import argparse


def parse_kgat_args():
    parser = argparse.ArgumentParser(description="Run KGAT")

    parser.add_argument("--seed", type=int, default=24, help="Random seed")

    parser.add_argument(
        "--data_name", nargs="?", default="melon", help="Choose dataset"
    )
    parser.add_argument(
        "--data_dir", nargs="?", default="datasets/", help="Input data path"
    )

    parser.add_argument(
        "--n_user", type=int, default=1000, help="Number of users to use"
    )
    parser.add_argument(
        "--nlp",
        type=str,
        default="vanilla",
        help="Choose nlp method for side information from {vanilla, keyword, similarity, embedding}",
    )

    parser.add_argument(
        "--use_embedding",
        type=int,
        default=0,
        help="0: No entity BERT embedding, 1: Use entity BERT embedding",
    )
    parser.add_argument(
        "--data_size",
        type=str,
        default="normal",
        help="Choose data size from {normal, small, big}",
    )

    parser.add_argument("--cf_batch_size", type=int, default=128, help="CF batch size")
    parser.add_argument("--kg_batch_size", type=int, default=128, help="KG batch size")
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1000,
        help="Test batch size (the user number to test every batch)",
    )

    parser.add_argument(
        "--embed_dim", type=int, default=256, help="User / entity Embedding size"
    )
    parser.add_argument(
        "--relation_dim", type=int, default=64, help="Relation Embedding size"
    )

    parser.add_argument(
        "--laplacian_type",
        type=str,
        default="random-walk",
        help="Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}",
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="bi-interaction",
        help="Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}",
    )
    parser.add_argument(
        "--conv_dim_list",
        nargs="?",
        default="[64, 32, 16]",
        help="Output sizes of every aggregation layer",
    )
    parser.add_argument(
        "--mess_dropout",
        nargs="?",
        default="[0.1, 0.1, 0.1]",
        help="Dropout probability w.r.t. message dropout for each deep layer (0: no dropout)",
    )

    parser.add_argument(
        "--kg_l2loss_lambda",
        type=float,
        default=1e-5,
        help="Lambda when calculating KG l2 loss",
    )
    parser.add_argument(
        "--cf_l2loss_lambda",
        type=float,
        default=1e-5,
        help="Lambda when calculating CF l2 loss",
    )

    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--n_epoch", type=int, default=200, help="Number of epoch")
    parser.add_argument(
        "--stopping_steps",
        type=int,
        default=10,
        help="Number of epoch for early stopping",
    )

    parser.add_argument(
        "--cf_print_every",
        type=int,
        default=1,
        help="Iter interval of printing CF loss",
    )
    parser.add_argument(
        "--kg_print_every",
        type=int,
        default=1,
        help="Iter interval of printing KG loss",
    )
    parser.add_argument(
        "--evaluate_every", type=int, default=10, help="Epoch interval of evaluating CF"
    )

    parser.add_argument(
        "--Ks",
        nargs="?",
        type=str,
        default="[1]+list(range(10,101,10))",
        help="Calculate metric@K when evaluating",
    )

    args = parser.parse_args()
    save_dir = f"trained_model/KGAT/{args.data_name}/user{args.n_user}_{args.nlp}_{args.use_embedding}_{args.data_size}_epoch{args.n_epoch}/"
    args.save_dir = save_dir

    return args
