import os
import sys
import argparse

sys.path.append("..")

import torch

from paths import (
    yangjie_rich_pretrain_unigram_path,
    yangjie_rich_pretrain_bigram_path,
    yangjie_rich_pretrain_word_path,
    yangjie_rich_pretrain_char_and_word_path,
    cmeee_v2_big_ner_path,
)

from load_data import load_cmeeev2_big_ner, load_yangjie_rich_pretrain_word_list
from modeling.add_lattice import equip_chinese_ner_with_lexicon
from fastNLP_module import BertEmbedding
from modeling.models import Lattice_Transformer_SeqLabel


def _move_to_device(batch_x, device):
    for k, v in list(batch_x.items()):
        if torch.is_tensor(v):
            batch_x[k] = v.to(device)
    return batch_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, type=str, help='Path to model .bin checkpoint (state_dict)')
    parser.add_argument('--output', default=None, type=str, help='Output .conll path (token\tpred_tag per line)')
    parser.add_argument('--device', default='0', type=str, help="cuda id like '0' or 'cpu'")
    parser.add_argument('--batch', default=16, type=int)

    # Must match training architecture
    parser.add_argument('--head_dim', default=20, type=int)
    parser.add_argument('--head', default=10, type=int)
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--bert_base', default='cn-wwm', type=str)

    # Data / lexicon config (keep aligned with training defaults)
    parser.add_argument('--number_normalized', default=0, type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--lattice_min_freq', default=1, type=int)
    parser.add_argument('--only_train_min_freq', default=True)
    parser.add_argument('--only_lexicon_in_train', default=False)
    parser.add_argument('--lexicon_name', default='yj', choices=['lk', 'yj'])

    args = parser.parse_args()

    if args.device != 'cpu':
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')

    if args.output is None:
        out_dir = os.path.dirname(os.path.abspath(args.ckpt))
        args.output = os.path.join(out_dir, 'CMeEE-V2_test.pred.conll')

    # 1) Load raw dataset (major-category version)
    datasets, vocabs, embeddings, _ = load_cmeeev2_big_ner(
        cmeee_v2_big_ner_path,
        yangjie_rich_pretrain_unigram_path,
        yangjie_rich_pretrain_bigram_path,
        index_token=False,
        _refresh=False,
        _cache_fp=os.path.join('/tmp', 'cmeeev2_big_pred_cache')
    )

    # 2) Lexicon augmentation (same pipeline as training)
    w_list = load_yangjie_rich_pretrain_word_list(
        yangjie_rich_pretrain_word_path,
        _refresh=False,
        _cache_fp='../cache/{}'.format(args.lexicon_name)
    )

    cache_name = os.path.join(
        '../cache',
        (args.lexicon_name + '_cmeee_v2_big_lattice' +
         '_only_train:{}' + '_norm_num:{}' +
         'lattice_min_freq{}' + 'only_train_min_freq{}' +
         'number_norm{}')
        .format(args.only_lexicon_in_train, args.number_normalized,
                args.lattice_min_freq, args.only_train_min_freq, args.number_normalized)
    )

    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(
        datasets, vocabs, embeddings,
        w_list, yangjie_rich_pretrain_word_path,
        _refresh=False, _cache_fp=cache_name,
        only_lexicon_in_train=args.only_lexicon_in_train,
        word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
        number_normalized=args.number_normalized,
        lattice_min_freq=args.lattice_min_freq,
        only_train_min_freq=args.only_train_min_freq
    )

    test_set = datasets['dev']

    # 3) Configure inputs for inference (no target)
    test_set.set_input('lattice', 'bigrams', 'seq_len')
    test_set.set_input('lex_num', 'pos_s', 'pos_e')
    test_set.set_pad_val('lattice', vocabs['lattice'].padding_idx)

    # Use dev max_seq_len for batching, but model relative-position table size must
    # match the checkpoint (otherwise load_state_dict will raise size mismatch).
    max_seq_len = int(max(test_set['seq_len']))

    # Load checkpoint early to infer rel-pos max_seq_len from its pe table.
    ckpt_state = torch.load(args.ckpt, map_location='cpu')
    if not isinstance(ckpt_state, dict):
        raise RuntimeError(f"Checkpoint must be a state_dict dict, got: {type(ckpt_state)}")

    ckpt_pe_key = None
    for _k in ('pe', 'encoder.pe', 'encoder.four_pos_fusion_embedding.pe'):
        if _k in ckpt_state:
            ckpt_pe_key = _k
            break
    if ckpt_pe_key is not None and torch.is_tensor(ckpt_state[ckpt_pe_key]):
        pe_len = int(ckpt_state[ckpt_pe_key].shape[0])
        # pe table length = 2*max_seq_len + 1
        ckpt_max_seq_len = (pe_len - 1) // 2
        if ckpt_max_seq_len > max_seq_len:
            max_seq_len = ckpt_max_seq_len

    # 4) Build model with training-like defaults
    hidden = args.head_dim * args.head
    ff_size = hidden * 3

    dropout = {
        'embed': 0.5,
        'gaz': 0.5,
        'output': 0.3,
        'pre': 0.5,
        'post': 0.3,
        'ff': 0.15,
        'ff_2': 0.15,
        'attn': 0.0,
    }
    mode = {'debug': 0, 'gpumm': False}

    bert_embedding = BertEmbedding(
        vocabs['lattice'],
        model_dir_or_name=args.bert_base,
        requires_grad=False,
        auto_truncate=True,
        word_dropout=0.01
    )

    model = Lattice_Transformer_SeqLabel(
        embeddings['lattice'],
        embeddings['bigram'],
        hidden,
        len(vocabs['label']),
        args.head,
        args.layer,
        use_abs_pos=False,
        use_rel_pos=True,
        learnable_position=False,
        add_position=False,
        layer_preprocess_sequence='',
        layer_postprocess_sequence='an',
        ff_size=ff_size,
        scaled=False,
        dropout=dropout,
        use_bigram=True,
        mode=mode,
        dvc=device,
        vocabs=vocabs,
        max_seq_len=max_seq_len,
        rel_pos_shared=True,
        k_proj=False,
        q_proj=True,
        v_proj=True,
        r_proj=True,
        self_supervised=False,
        attn_ff=False,
        pos_norm=False,
        ff_activate='relu',
        abs_pos_fusion_func='nonlinear_add',
        embed_dropout_pos='0',
        four_pos_shared=True,
        four_pos_fusion='ff_two',
        four_pos_fusion_shared=True,
        bert_embedding=bert_embedding,
        is_ctr=False,
        id_to_label=None,
        temp=0.07,
        only_head=False,
        hard_k=0,
        hard_weight=0.0,
        disease_ids=None,
        other_ids=None,
        lambda_hsr=0.0,
        cf_lambda=0.0,
    )

    # Load weights (ckpt_state is on CPU; PyTorch will copy to model device).
    model.load_state_dict(ckpt_state, strict=False)
    model.to(device)
    model.eval()

    # 5) Iterate test set and write predictions
    from fastNLP.core.batch import DataSetIter
    from fastNLP.core.sampler import SequentialSampler

    label_vocab = vocabs['label'].idx2word

    data_iter = DataSetIter(dataset=test_set, batch_size=args.batch, sampler=SequentialSampler())

    with open(args.output, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for batch_x, _ in data_iter:
                indices = data_iter.get_batch_indices()
                batch_x = _move_to_device(batch_x, device)

                # Model forward() requires `target` in signature even in eval mode.
                # For pure inference, pass target=None.
                if 'target' in batch_x:
                    pred_dict = model(**batch_x)
                else:
                    pred_dict = model(target=None, **batch_x)
                pred = pred_dict['pred']
                seq_len = batch_x['seq_len']

                for bi, idx in enumerate(indices):
                    chars = test_set[idx]['raw_chars']
                    sl = int(seq_len[bi])
                    pred_ids = pred[bi][:sl].tolist()
                    # raw_chars length should equal sl (character part). Use raw_chars to be safe.
                    for ch, pid in zip(chars, pred_ids[:len(chars)]):
                        f.write(f"{ch}\t{label_vocab[int(pid)]}\n")
                    f.write("\n")

    print(f"Saved predictions to: {args.output}")


if __name__ == '__main__':
    main()
