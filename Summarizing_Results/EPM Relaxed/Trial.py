import pandas as pd

no_dist=[2,10,30,40,50,100] # the number of districts
probs_list=['CARP_F15_g_graph.dat','CARP_N17_g_graph.dat','CARP_K13_g_graph.dat','CARP_O12_g_graph.dat','CARP_S9_p_graph.dat','CARP_N16_g_graph.dat','CARP_S9_g_graph.dat','CARP_S11_g_graph.dat','CARP_K9_p_graph.dat','CARP_N9_g_graph.dat','CARP_N11_g_graph.dat','CARP_N15_g_graph.dat']

target_row_index = 2  # Adjust this index as needed
target_column_name = 0
# for prob in probs_list:
#     for num in no_dist:
#         file_path = 'EPM_no_dist_' + str(num) + '_prob_' + str(prob) + '.csv'
#
#         # Read the CSV file into a DataFrame
#         df = pd.read_csv(file_path,delimiter=',')
#
#         # Change the value in the specified row and column
#         df.iat[target_row_index, target_column_name] = 'Computation Time_EPM'
#
#         # Save the modified DataFrame back to the CSV file
#         df.to_csv(file_path, index=False)

# Read the CSV file and modify the specific cell
with open(file_path, 'r', newline='') as infile, open(output_path, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for i, row in enumerate(reader):
        if i == row_index:
            try:
                row[column_index] = new_value
            except IndexError:
                print(f"IndexError: Row {row_index} doesn't have enough columns.")
        writer.writerow(row)