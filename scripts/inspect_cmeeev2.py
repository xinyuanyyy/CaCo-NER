import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from load_data import load_cmeeev2_ner


cmeee_ner_path = r"/home/u2025170862/jupyterlab/nerco/data/datasets/CMeEE-V2-Resplit-CoNLL"
PRETRAINED_CHAR = r"/home/u2025170862/jupyterlab/nerco/data/word/gigaword_chn.all.a2b.uni.ite50.vec"
PRETRAINED_BIGRAM = r"/home/u2025170862/jupyterlab/nerco/data/word/gigaword_chn.all.a2b.bi.ite50.vec"

def _describe_embedding(emb):
    if emb is None:
        return "<None>"
    try:
        shape = tuple(emb.embedding.weight.shape)
    except AttributeError:
        try:
            shape = tuple(emb.weight.shape)
        except AttributeError:
            shape = "<unknown>"
    return f"{type(emb).__name__} shape={shape}"


def main() -> None:
    datasets, vocabs, embeddings, id_to_label = load_cmeeev2_ner(
        cmeee_ner_path,
        unigram_embedding_path=PRETRAINED_CHAR,
        bigram_embedding_path=PRETRAINED_BIGRAM,
        index_token=True,
        _refresh=True,
    )

    print("DATASET SIZES:")
    for split in ("train", "dev", "test"):
        print(f"  {split}: {len(datasets[split])}")
    print()

    print("VOCABULARIES:")
    for key, vocab in vocabs.items():
        print(f"  {key}: size={len(vocab)}")
    print()

    print("EMBEDDINGS:")
    for key, emb in embeddings.items():
        print(f"  {key}: {_describe_embedding(emb)}")
    print()

    print("LABEL MAPPING (id_to_label):")
    print(id_to_label)
    print()

    print("SAMPLE INSTANCE:")
    example = datasets["train"][0]

    # 尝试还原原始文本和标签
    try:
        char_vocab = vocabs['char']
        # fastNLP Vocabulary use to_word(idx)
        raw_chars = [char_vocab.to_word(idx) for idx in example['chars']]
        print(f"  [Reconstructed Text]: {''.join(raw_chars)}")
        
        raw_labels = [id_to_label[idx] for idx in example['target']]
        print(f"  [Reconstructed Labels]: {raw_labels}")
    except Exception as e:
        print(f"  [Reconstruction Failed]: {e}")

    print("-" * 30)
    print("  [Processed Fields (IDs/Tensors)]:")
    for field_name in example.fields:
        value = example[field_name]
        # Show first 20 items for lists to give a better view
        preview = value[:20] if hasattr(value, "__len__") else value
        print(f"    {field_name}: {preview}")


if __name__ == "__main__":
    main()
