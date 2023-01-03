import argparse
import os
import glob
import csv


def main(args):

    for dir, file in zip(args.image_dirs, args.target_csv_files):
        with open(file, 'w', newline='') as csvfile:
            fieldnames = ['img_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            images = glob.glob(os.path.join(dir, "*.jpg"))
            for image in images:
                writer.writerow({'img_path': image})

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Prep csv file for fairface prediction')
    parser.add_argument('--image_dirs', 
                        type=str, 
                        nargs = "+", 
                        required=True,
                        help='directories whose images to make fairface csvs for')
    parser.add_argument('--target_csv_files', 
                        type=str, 
                        nargs = "+", 
                        required=True,
                        help='Filename for fairface csvs for')
        
    main_args = args.parse_args()

    if len(main_args.image_dirs) != len(main_args.target_csv_files):
        raise Exception("Num image dirs must match num target csv files")
    main(main_args)