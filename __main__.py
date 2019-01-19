import os
import argparse
import json
import csv
from src.utils.image_utils import load_image
from src.ea import evolutionary_algorithm as evol


def check_invalid_args(args):
    if args.grayscale and args.original is not None:
        print("The combination of '-o' and '-g' is not supported.")
        exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local_models", help="Use local models for fooling",
                        action="store_true")
    parser.add_argument("-o", "--original", help="Input image as startup aid")
    parser.add_argument("-p", "--poly", help="Use polygons instead of random pixels",
                        action='store_true')
    parser.add_argument("-c", "--classes", nargs='+', type=int,
                        help="Space separated class label list, indicating for" +
                             "which classes a fooling image should be produced")
    parser.add_argument("-g", "--grayscale", help="Generate grayscale images", action="store_true")
    parser.add_argument("-s", "--statistic", help="Save statistic to file", action="store_true")
    args = parser.parse_args()
    check_invalid_args(args)

    original = None
    class_index = [0]
    dpath = os.path.dirname(__file__)

    if not os.path.isdir(os.path.join(dpath, "results/")):
        os.makedirs(os.path.join(dpath, "results/"))

    if args.original is not None:
        original = load_image(args.original, 64)

    if args.classes is not None:
        class_index = args.classes

    # get the lists of paths for local models
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:

        # load model paths from config and check paths for correctness
        data = json.load(f)
        models = data["local_models"]
        if len(models) == 0 and args.local_models:
            raise Exception("No local model specified in config file")
        models = [os.path.join(dpath, m) for m in models]
        for model_path in models:
            if not os.path.exists(model_path):
                raise FileNotFoundError("Local model file '" + model_path + "' does not exist.")

        # parse class index label dict
        index_label_path = os.path.join(dpath, data["class_label_list"])
        with open(index_label_path, "r", encoding="utf8") as csv_file:
            reader = csv.reader(csv_file)
            index_to_label = {rows[0]: rows[1] for rows in reader}

        # check input class index
        for index in class_index:
            if index < -1 or index >= len(index_to_label):
                raise (ValueError("Class index must be between 0 and {}".format(len(
                    index_to_label))))

        # parse all other ea parameters
        ea_params = data["ea_params_other"]

    if args.grayscale:
        color_range = 1
    else:
        color_range = 3

    ea = evol.EvolutionaryAlgorithm(class_index, models, index_to_label, ea_params,
                                    color_range, args.local_models, args.poly, original)

    for index in class_index:
        ea.class_index = index
        ea.run()
        if args.statistic:
            ea.print_statistic()
    print("Results can be viewed in folder 'informaticup2019-LabWITT-EA/results/'")


if __name__ == '__main__':
    main()
