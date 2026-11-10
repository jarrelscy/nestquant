"""Full-vocabulary KL(teacher || student) per prediction position, computed in float64."""

import numpy as np


def _log_softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def tokenwise_kld(teacher_logits, student_logits):
    teacher = np.asarray(teacher_logits, dtype=np.float64)
    student = np.asarray(student_logits, dtype=np.float64)
    if teacher.ndim != 2 or teacher.shape != student.shape:
        raise ValueError('Expected two logit arrays of identical [positions, vocabulary] shape')
    if not (np.isfinite(teacher).all() and np.isfinite(student).all()):
        raise ValueError('Logits must be finite')
    teacher_log = _log_softmax(teacher)
    return (np.exp(teacher_log) * (teacher_log - _log_softmax(student))).sum(axis=-1)


def summarize(kld, teacher_logits, student_logits):
    kld = np.asarray(kld, dtype=np.float64)
    agreement = np.asarray(teacher_logits).argmax(axis=-1) == np.asarray(student_logits).argmax(axis=-1)
    return dict(count=int(kld.size), mean=float(kld.mean()), p50=float(np.quantile(kld, 0.5)),
                p95=float(np.quantile(kld, 0.95)), p99=float(np.quantile(kld, 0.99)), max=float(kld.max()),
                top1_agreement=float(agreement.mean()))
