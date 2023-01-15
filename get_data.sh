# !/bin/bash

# # get train and validation data for affect net (small version)
# mkdir affectnet
# wget -O affectnet/train_set.tar https://public.dm.files.1drv.com/y4mYit9LKGngVKReCv3WHU9DxTYhKN72l89KnWJ1Vg8Ia7PbAVAZZIpVwe4vH9Z2W1Z6e6_8wt66ehCRbHd2ei7Wad7mWDbjy8DKM8lltwL-sZFIIRdYK7JN7dvFBL0BST29qyophyzu_mGIk4zf3M-o9Efx7BgYXEoGxjLxtOKVta44N-uUe3vQ-R0t5o30zw0c93I1TT-QOdPvkyvnEglb-VqmzKwFDOO79E5fjX83gY
# wget -O affectnet/val_set.tar https://public.dm.files.1drv.com/y4mvhN0bHm5KUYZrBgoNVnAvuDBXabGMpgShdCF9EsyB35JoZ5q_tncTOETy4JyxUlt0yIVcgWJUaS1YqESHI7wgcKZa_9qNaVxt6DZR0wHlDkMqw5Ejv0tnyK7hyuHXKvX04w-MdqawyKeEE04QuQijcJHX3EMn-SNpnp1Frw4ecwupatjM9OYcoSva_vSImqo4T8nRSvJcYlSJHa4pZeGi9BOuU9qpauxN5FC6VjiwYQ

# we'll just manually scp over the data for now
# unzip tars
tar -xvf ../train_set.tar
tar -xvf ../val_set.tar

# get the fairface csvs ready
python prep_fairface_data.py --image_dirs train_set/images val_set/images --target_csv_files train_fairface.csv val_fairface.csv

fairface_file_name="./FairFace.tar"

# check fairface exists
if [ ! -f "$fairface_file_name" ]
then
    echo "Unable to find FairFace tar file; download from the FairFace Github and modify w/ appropriate models"
    exit
fi

tar -xvf ./FairFace.tar

cd FairFace

python ./predict.py --csv ../val_fairface.csv