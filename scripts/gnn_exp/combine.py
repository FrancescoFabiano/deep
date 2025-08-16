import os

from scripts.gnn_exp.generate_summary_tables import generate_latex_table
from scripts.gnn_exp.plot_scaling import main_plot_scaling


def combine_tex_files(input_files, output_file):
    """
    Combine multiple .tex files into one, adding a blank line between each.

    Args:
        input_files (List[str]): Paths to .tex files to combine.
        output_file (str): Path to the resulting combined .tex file.
    """
    with open(output_file, "w") as outfile:
        for i, file_path in enumerate(input_files):
            if not os.path.isfile(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            with open(file_path, "r") as infile:
                content = infile.read()
                outfile.write(content.rstrip())  # avoid extra trailing space
                if i != len(input_files) - 1:
                    outfile.write("\n\n")  # empty line between files
        print(f"Combined {len(input_files)} files into '{output_file}'")


def get_names(file_res:str="../../exp/gnn_exp/final_results_ok"):
    batches = {
        "batch1": ["Assemble", "CC", "CoinBox", "Grapevine", "SC"],
        #"batch1_refined": ["CC_2_2_4","CC_3_3_3","SC_Four_Rooms","SC_Multi_Rooms"],
        "batch2": ["CC_2_2_4__pl_7", "CC_3_2_3__pl_6", "CoinBox_pl__3", "CoinBox_pl__5"],
        "batch3":["SCRich"],
        "batch4": ["All", "CC-CoinBox", "CC-CoinBox-Grapevine", "CC-Grapevine"],
        "batch4_partial": ["CC-CoinBox", "CC-CoinBox-Grapevine", "CC-Grapevine"],
    }

    os.makedirs(file_res, exist_ok=True)

    path_list = []
    for batch_n, list_of_domains in batches.items():

        DOMAIN_TRIPLES = []
        for domain in list_of_domains:
            output_file_train = f"{file_res}/{batch_n}/{domain}_comparison_train.tex"
            output_file_test = f"{file_res}/{batch_n}/{domain}_comparison_test.tex"

            DOMAIN_TRIPLES.append((output_file_train,output_file_test, domain))

            path_list.append(output_file_train)
            path_list.append(output_file_test)

            if batch_n == "batch5" or batch_n == "batch5_partial":
                n = 3
            else:
                n = 2

            title_train = f"Scalability {batch_n.replace("b", "B")} - {domain} - Train"
            figname_train = f"{file_res}/{batch_n}/{domain}_scalability_comparison_train.pdf"
            main_plot_scaling(output_file_train, n, title_train, figname_train)

            title_test = f"Scalability on {domain} - Test"
            figname_test = f"{file_res}/{batch_n}/{domain}_scalability_comparison_test.pdf"
            main_plot_scaling(output_file_test, n, title_test, figname_test)

        label = f"tab:{batch_n}_res"
        caption = f"{batch_n} summary iqm"
        outputfile = f"{file_res}/{batch_n}/{batch_n}_summary_comparison.tex"
        generate_latex_table(DOMAIN_TRIPLES, caption, label, outputfile)

    return path_list

# Example usage
if __name__ == "__main__":
    input_files = get_names()
    output_file = "../../exp/gnn_exp/reports/all_results.tex"
    combine_tex_files(input_files, output_file)