"""Grounding DINO, queried under several phrasings at once.

The detector is used as published: no fine-tuning, no adapters, the released
``IDEA-Research/grounding-dino-tiny`` weights.  Everything this module does is arrange
the text side of the call so that one image can be scored against five wordings, and
read the per-phrase scores back out of the token-aligned logits.

Why the five wordings are five forwards
---------------------------------------
The backbone does not see the text: it encodes the image once and the text encoder
produces the phrase embeddings.  Batching the five queries into one call therefore
repeats the *pixel* tensor five times and makes the model encode the same image five
times inside that call, which costs exactly the same arithmetic as five separate calls
but five times the activation memory.  On LVIS that difference was between a run that
fits in 24 GB and one that spills into host memory and stalls.

Token alignment
---------------
Grounding DINO scores each box against a set of token spans, one per phrase, and the
score of a phrase is the maximum over its tokens.  ``spans_for`` builds the input ids
``[CLS] w w w . [SEP]`` and records where each phrase starts and ends, so the score of
"a photo of a dog" is read from the tokens of that phrase only, not from the whole
sentence.
"""
import numpy as np
import torch

from ..prompts import VARIANT_NAMES, variants

MODEL = "IDEA-Research/grounding-dino-tiny"

CLS, SEP, DOT, PAD = 101, 102, 1012, 0


def load_model(device="cuda"):
    """Processor and model.  The checkpoint is downloaded on first use."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL)
    return proc, model.to(device).eval()


def spans_for(phrase_ids):
    """``[CLS] p1 . p2 . ... [SEP]`` and the (start, end) span of every phrase."""
    ids = [CLS]
    spans = []
    for t in phrase_ids:
        s = len(ids)
        ids.extend(t)
        spans.append((s, len(ids)))
        ids.append(DOT)
    ids.append(SEP)
    return ids, spans


@torch.inference_mode()
def run_image(model, proc, img_rgb, phrase_lists, device="cuda"):
    """Score one image against every wording.

    ``phrase_lists[v]`` is the list of category phrases for wording ``v``; one text
    query per wording is what a user typing that wording would produce.

    Returns ``(boxes, scores, enc)`` where ``boxes`` is ``(V, Q, 4)`` in pixel xyxy,
    ``scores[v][k]`` is the ``(Q,)`` score of phrase ``k`` under wording ``v`` and
    ``enc[v]`` is the tokenisation, kept so the caller can recompute spans.
    """
    tok = proc.tokenizer
    enc = [spans_for([tok(p, add_special_tokens=False)["input_ids"] for p in pl])
           for pl in phrase_lists]
    texts = [" ".join(pl) for pl in phrase_lists]
    L = max(len(e[0]) for e in enc)
    ids = torch.full((len(texts), L), PAD, dtype=torch.long)
    att = torch.zeros((len(texts), L), dtype=torch.long)
    for i, (seq, _) in enumerate(enc):
        ids[i, :len(seq)] = torch.tensor(seq)
        att[i, :len(seq)] = 1
    pix = proc.image_processor(images=img_rgb, return_tensors="pt")["pixel_values"]
    pix = pix.to(device)
    boxes_all, sc = [], []
    for i in range(len(texts)):
        out = model(pixel_values=pix, input_ids=ids[i:i + 1].to(device),
                    attention_mask=att[i:i + 1].to(device),
                    token_type_ids=torch.zeros_like(ids[i:i + 1], device=device))
        boxes_all.append(out.pred_boxes)
        prob = out.logits.sigmoid()
        sc.append([prob[0, :, s:e].amax(dim=1) for s, e in enc[i][1]])
    boxes = torch.cat(boxes_all, 0)
    h, w = img_rgb.shape[:2]
    cx, cy, bw, bh = boxes.unbind(-1)
    xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h], dim=-1)
    return xyxy.cpu().numpy(), [[v.cpu().numpy() for v in row] for row in sc], enc


def phrase_keys(names, cat_ids):
    """Every distinct phrase string, in a stable order, without duplicates.

    A phrase is shared by many detections --- "a dog" is one string whatever the image
    --- so the text embedding is computed once per distinct string, which is what keeps
    the descriptor cheap: a corpus has a few thousand phrases and millions of detections.
    """
    seen, keys = set(), []
    for c in cat_ids:
        for s in variants(names[c]):
            if s not in seen:
                seen.add(s)
                keys.append(s)
    return keys


@torch.inference_mode()
def embed_phrases(model, proc, keys, device="cuda", batch=64):
    """Pooled last-hidden-state embedding of every phrase, ``(P, 768)`` float16.

    Mean pooling over the tokens with the attention mask applied: the descriptor of a
    phrase has to depend on the whole sentence, not on the token the detector happens
    to score highest.
    """
    tok = proc.tokenizer
    out = []
    for i in range(0, len(keys), batch):
        enc = tok(keys[i:i + batch], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask", "token_type_ids")}
        h = model.model.text_backbone(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def variant_phrases(names, cat_ids):
    """The phrase lists for one image: ``[ [wording v for every category] for v ]``."""
    return [[variants(names[c])[v] for c in cat_ids] for v in range(len(VARIANT_NAMES))]
