import numpy as np
import torch
import numbers

def select_action(action_out):
    log_p_a = action_out
    p_a = [[z.exp() for z in x] for x in log_p_a]
    ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
    return ret

def translate_action(action):
    action = [x.squeeze().data.numpy() for x in action]
    actual = action                     # attention here : check the tyoe of actual.
    return action, actual

def multinomials_log_densities(actions, log_probs):
    log_prob = [0] * len(log_probs)
    for i in range(len(log_probs)):
        log_prob[i] += log_probs[i].gather(1, actions[:, i].long().squeeze(1))
    log_prob = torch.cat(log_prob, dim=-1)
    return log_prob

def multinomials_log_density(actions, log_probs):
    log_prob = 0
    for i in range(len(log_probs)):
        log_prob += log_probs[i].gather(1, actions[:, i].long().unsqueeze(1))
    return log_prob

def merge_stat(src, dest):
    for k, v in src.items():
        if not k in dest:
            dest[k] = v
        elif isinstance(v, numbers.Number):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, np.ndarray): # for rewards in case of multi-agent
            dest[k] = dest.get(k, 0) + v
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]