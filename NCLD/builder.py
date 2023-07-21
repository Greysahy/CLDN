import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SSFL(nn.Module):
    """
    Self Supervised guided Feature Learner
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, class_num=1000):
        super(SSFL, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        dim_mlp = 4096
        self.q_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        self.k_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
        
        self.classifier = nn.Linear(dim_mlp, class_num)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("label_queue", torch.zeros(K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        for param_q, param_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.size(0)

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.label_queue[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        if ptr + batch_size > self.queue.shape[1]:
            ptr = self.queue.shape[1] - batch_size
        
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size = x.shape[0]
        # random shuffle index
        idx_shuffle = torch.randperm(batch_size).to(x.device)
        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    def forward(self, im_q, im_k=None, labels=None):
        # compute query features
        out_q = self.encoder_q(im_q)  # queries: NxC
        pred = self.classifier(out_q)

        if not self.training:
            return pred
        
        q = self.q_fc(out_q)
        q = nn.functional.normalize(q, dim=1)
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            out_k = self.encoder_k(im_k)  # keys: NxC
            k = self.k_fc(out_k)
            k = nn.functional.normalize(k, dim=1)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        contrastive_loss = self.cal_contrastive_loss(q, k, labels)
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, labels)
        
        return pred, contrastive_loss
    
    def cal_contrastive_loss(self, q, k, labels):
        bsz = q.shape[0]
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits_pos = l_pos / self.T
        exp_logits_pos = torch.exp(logits_pos)
        logits_neg = l_neg / self.T
        exp_logits_neg = torch.exp(logits_neg)
        
        q_labels_expand = labels.reshape(-1, 1).expand(bsz, self.K).long()
        k_labels_expand = self.label_queue.reshape(1, -1).expand(bsz, self.K).long()
        mask = 1 - (q_labels_expand == k_labels_expand).float()
        exp_logits_neg = exp_logits_neg * mask
        
        exp_logits = torch.cat([exp_logits_pos, exp_logits_neg], dim=1)
        contrastive_loss = -(logits_pos - torch.log(exp_logits.sum(1, keepdim=True))).mean()

        return contrastive_loss

    
class SDL(nn.Module):
    """
    Self Distillation Learner
    """
    def __init__(self, teacher_encoder,student_encoder, classifier_t, class_num, teacher_ckpt=None):
        super(SDL, self).__init__()
        
        self.encoder_q = student_encoder()
        self.encoder_t = teacher_encoder()  # teacher
        
        dim_mlp = 4096
        self.classifier = nn.Linear(dim_mlp, class_num)
        self.classifier_kd = nn.Linear(dim_mlp, class_num)
        self.classifier_t = classifier_t(num_classes=class_num, feat_dim=4096)

        for param in self.encoder_t.parameters():
            param.requires_grad = False
        for param in self.classifier_t.parameters():
            param.requires_grad = False
        
        if teacher_ckpt is not None:
            self.load_teacher(teacher_ckpt)

    
    def load_teacher(self, weight_path):
        assert weight_path is not None
        print('=> load teacher model weights from {}'.format(weight_path))
        state_dict = torch.load(weight_path, map_location='cpu')
        encoder_weights = OrderedDict()
        classifier_weights = OrderedDict()

        for k, v in state_dict.items():
            if 'encoder_q' in k:
                encoder_weights[k.replace('encoder_q.', '')] = v
            elif 'classifier' in k:
                classifier_weights[k.replace('classifier.', '')] = v
        self.encoder_t.load_state_dict(encoder_weights, strict=False)
        self.classifier_t.load_state_dict(classifier_weights, strict=False)
    
    def forward(self, img, label, t):
        out_q = self.encoder_q(img)
        pred = self.classifier(out_q)
        pred_kd = self.classifier_kd(out_q)
        
        if not self.training:
            return pred, pred_kd

        with torch.no_grad():
            out_t = self.encoder_t(img)
            pred_t = self.classifier_t(out_t)
        
        loss_kd, pred_teacher_part2 = self.nkl_loss(logits_student=pred_kd, logits_teacher=pred_t.detach(), target=label, temperature=t)
#         loss_kd = self.kl_loss(pred_kd, pred_t.detach())
        return pred, pred_kd, loss_kd
    
    def kl_loss(self, pred_s, pred_t):
        p_s = F.log_softmax(pred_s / self.T, dim=1)
        p_t = F.softmax(pred_t / self.T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / p_s.shape[0]
        return loss_kd
    
    def nkl_loss(self, logits_student, logits_teacher, target, temperature):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nkl_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature**2)
            / target.shape[0]
        )
        return nkl_loss, pred_teacher_part2


    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1).to(torch.int64), 1).bool()
        return mask


    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1).to(torch.int64), 0).bool()
        return mask


    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
