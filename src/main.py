"""Running the Splitter."""

import torch
from param_parser import parameter_parser
from splitter import SplitterTrainer
from utils import tab_printer, graph_reader
from visualize import visualize_embeddings_comparison

def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    graph = graph_reader(args.edge_path)
    trainer = SplitterTrainer(graph, args)
    trainer.fit()
    trainer.save_embedding()
    trainer.save_persona_graph_mapping()
    visualize_embeddings_comparison(
        base_path="./output/chameleon_base_embedding.csv",
        final_path=args.embedding_output_path,
        label_path="./input/chameleon_target.csv"
    )

if __name__ == "__main__":
    main()
