import csv
import os

csv_files = "input-data/initial-csv-files/female/"

for root, dirs, files in os.walk(csv_files):
    for csv_file in files:
        file_name = os.path.join(root, csv_file)
        temp_name = os.path.join(root, "temp.csv")

        if ".DS_Store" in file_name or "temp.csv" in file_name:
            continue

        with open(file_name,'rU') as f:
            reader = list(csv.reader(f, delimiter=','))
            row_count = len(reader)
            with open(temp_name, 'w+') as temp:
                writer = csv.writer(temp, delimiter=',')

                for i, row in enumerate(reader):
                    if i == row_count - 1:
                        continue
                    else:
                        writer.writerow(row)

                temp.seek(0)
                with open(file_name,"w") as out:
                    reader = csv.reader(temp , delimiter = ",")
                    writer = csv.writer(out, delimiter = ",")
                    for row in reader:
                        writer.writerow(row)