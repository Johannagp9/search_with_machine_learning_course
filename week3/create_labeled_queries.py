import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

def clean_text(text: str):
    text = text.lower()
    text = "".join([ c if (c.isalpha() or c.isnumeric()) else c for c in text ])
    text = text.strip()
    return stemmer.stem(text) 


parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
queries_df['query'] = queries_df['query'].apply(clean_text)

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
queries_df_count =  queries_df.groupby(['category']).size().reset_index(name='count')
queries_df_parent = pd.merge(queries_df_count, parents_df, left_on='category', right_on='category')
queries_df_parent = queries_df_parent.sort_values('count')

for idx, row in tqdm(queries_df_parent.iterrows()):
    if row['count'] < min_queries:
        queries_df.loc[queries_df['category'] == row['category'], 'category'] = row['parent']
        queries_df_count.loc[queries_df_count['category'] == row['parent'], 'count'] = len(queries_df['category']==row['parent'])
    else:
        break


# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
