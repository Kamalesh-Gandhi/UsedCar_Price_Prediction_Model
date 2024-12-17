import pandas as pd
import ast

# Function to read Excel files
def Read_CSV(FilePath):
    try:
        df = pd.read_excel(FilePath)
        print(f'The Excel file from {FilePath} got read successfully','\n')
        return df
    except Exception as e:
        print(f'Error While Reading the file from {FilePath} : ', e, '\n')

# Flatten 'new_car_detail' column
def process_new_car_detail(data):
    dataframe = []
    column = data['new_car_detail'].apply(ast.literal_eval)

    for cell in column:
        flattened_data = {**cell, **cell['trendingText']}
        flattened_data.pop('trendingText')
        dataframe.append(flattened_data)

    return pd.DataFrame(dataframe)

# Flatten 'new_car_overview' column
def process_new_car_overview(data):
    dataframe = []
    column = data['new_car_overview'].apply(ast.literal_eval)

    for cell in column:
        flattened_data = {item['key']: item['value'] for item in cell['top']}
        dataframe.append(flattened_data)

    return pd.DataFrame(dataframe)

# Flatten 'new_car_feature' column
def process_new_car_feature(data):
    dataframe = []
    column = data['new_car_feature'].apply(ast.literal_eval)

    for cell in column:
        flattened_data = {'Features': len(cell['top'])}
        for item in cell['data']:
            flattened_data[item['heading']] = len(item['list'])
        dataframe.append(flattened_data)

    return pd.DataFrame(dataframe)

# Flatten 'new_car_specs' column
def process_new_car_specs(data):
    dataframe = []
    column = data['new_car_specs'].apply(ast.literal_eval)

    for cell in column:
        flattened_data = {item['key']: item['value'] for item in cell['top']}
        for item in cell['data']:
            for val in item['list']:
                flattened_data[val['key']] = val['value']
        dataframe.append(flattened_data)

    return pd.DataFrame(dataframe)

# Function to combine all processed data
def process_and_combine_city_data(city, data):
    data1 = process_new_car_detail(data)
    data2 = process_new_car_overview(data)
    data3 = process_new_car_feature(data)
    data4 = process_new_car_specs(data)

    Complete_data = pd.concat([data1.reset_index(drop=True), 
                               data2.reset_index(drop=True), 
                               data3.reset_index(drop=True), 
                               data4.reset_index(drop=True)], axis=1)
    Complete_data.insert(0, 'city', city)
    return Complete_data

# File paths for different cities
filepath = [
    (r'DataSets\chennai_cars.xlsx', 'Chennai'),
    (r'DataSets\bangalore_cars.xlsx', 'Bangalore'),
    (r'DataSets\delhi_cars.xlsx', 'Delhi'),
    (r'DataSets\hyderabad_cars.xlsx', 'Hyderabad'),
    (r'DataSets\jaipur_cars.xlsx', 'Jaipur'),
    (r'DataSets\kolkata_cars.xlsx', 'Kolkata')
]

all_data = []

# Loop through each file and process data
for path, city in filepath:
    df = Read_CSV(path)  # Read the Excel file
    if df is not None:  # Proceed if the file was read successfully
        city_data = process_and_combine_city_data(city, df)  # Process and combine data
        all_data.append(city_data)
        # city_file_name = f'DataSets/Structured_{city}_cars.csv'  # Create filename for each city
        # city_data.to_csv(city_file_name, index=False)  # Save the processed data to CSV
        # print(f'Structured data for {city} saved as {city_file_name}','\n')
        # print(f'Structured data for {city} saved','\n')
        

for idx, df in enumerate(all_data):
    print(f'DataFrame {idx} shape: {df.shape}')
    print(f'DataFrame {idx} columns: {df.columns.tolist()}','\n')

# Check for any duplicate index or issues in the DataFrames
for idx, df in enumerate(all_data):
    if not df.index.is_unique:
        print(f"DataFrame {idx} has duplicate indices!")

print(f'All_Data{all_data}','\n')

# Reset index for each DataFrame to ensure uniqueness
all_data_reset = [df.reset_index(drop=True) for df in all_data]

# Combine all DataFrames row-wise 
try:
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    print('Combined all city data into one data file', '\n')
except Exception as e:
    print('Error while combining data: ', e)

# Save the combined data to a single CSV file
combined_data.to_csv('DataSets/Combined_Cars_Data.csv')

print('Combined all city datas into one data File','\n')