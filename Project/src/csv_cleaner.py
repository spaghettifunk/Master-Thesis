'''
The MIT License

Copyright (c) 2015 University of Rochester, Uppsala University
Authors: Davide Berdin, Philip J. Guo, Olle Galmo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

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
                    reader = csv.reader(temp , delimiter=",")
                    writer = csv.writer(out, delimiter=",")
                    for row in reader:
                        writer.writerow(row)