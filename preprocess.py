import os
import pandas as pd


def concatenate_columns(folder_path, text_columns, output_file_path):
    """
    Reads multiple CSV files from the given folder, extracts text from specified columns,
    removes duplicates and whitespace, and saves the combined text into a single output CSV.

    Parameters:
    - folder_path (str): Path to the folder containing input CSV files.
    - text_columns (list): List of column names to extract text from.
    - output_file_path (str): Path to save the output CSV file.
    """
    text_list = []

    # Loop over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, low_memory=False)
            df.dropna(inplace=True)
            print(f"Processing file: {filename}")

            # Extract text from the specified columns
            for column_name in text_columns:
                if column_name in df.columns:
                    text_list.extend(df[column_name].to_list())

    # Remove duplicates and strip extra whitespace
    text_list = list(set(text_list))
    cleaned_text_list = [sentence.replace("\n", "").strip() for sentence in text_list]

    # Save the cleaned list into a DataFrame
    output_df = pd.DataFrame({'text': cleaned_text_list})
    output_df.dropna(inplace=True)
    output_df.to_csv(output_file_path, index=False, encoding='utf-8')


def extract_vocab_from_csv(file_path):
    """
    Extracts a list of vocabulary words from a CSV file with a column 'Vocab'.

    Parameters:
    - file_path (str): Path to the CSV file containing vocabulary.

    Returns:
    - list: A list of individual vocabulary words.
    """
    vocab_words = []
    df = pd.read_csv(file_path)

    for sentence in df['Vocab']:
        sentence = sentence.replace(",", "")
        vocab_words.extend(sentence.split())

    return vocab_words


def filter_sentences_by_vocab(sentences, vocab_words):
    """
    Splits sentences into two lists based on whether they contain any vocabulary word.

    Parameters:
    - sentences (list): List of text sentences.
    - vocab_words (list): List of vocabulary words.

    Returns:
    - tuple: (matched_sentences, unmatched_sentences)
    """
    matched, unmatched = [], []

    for i, sentence in enumerate(sentences):
        print(f"Checking sentence {i}")
        if any(word in sentence.split() for word in vocab_words):
            matched.append(sentence)
        else:
            unmatched.append(sentence)

    return matched, unmatched


def main():
    # Step 1: Concatenate and clean text columns from multiple CSV files
    folder_path = r'D:\IIT PATNA\BTP\Dataset'
    columns_to_extract = ['text', 'tweet_text']
    combined_output_path = 'final_output_without_duplicates.csv'

    concatenate_columns(folder_path, columns_to_extract, combined_output_path)

    # Step 2: Extract initial vocabulary
    main_vocab_path = "Cat_with_Vocab_updated_final.csv"
    vocab_words = extract_vocab_from_csv(main_vocab_path)
    print("Initial vocab:", vocab_words)

    # Step 3: Load cleaned text data
    new_df = pd.read_csv(combined_output_path)

    # Step 4: Filter text based on vocabulary
    matched, unmatched = filter_sentences_by_vocab(new_df['text'], vocab_words)

    pd.DataFrame({'text': matched}).to_csv("matched_text_data_without_stem.csv", index=False, encoding='utf-8')
    pd.DataFrame({'text': unmatched}).to_csv("unmatched_text_data_without_stem.csv", index=False, encoding='utf-8')

    # Step 5: Extend vocabulary and repeat filtering
    extended_vocab_path = "Final_ext_vocab_wiki.csv"
    extended_vocab = extract_vocab_from_csv(extended_vocab_path)
    vocab_words.extend(extended_vocab)
    print("Extended vocab:", vocab_words)

    # Re-run filtering with extended vocab
    matched_ext, unmatched_ext = filter_sentences_by_vocab(new_df['text'], vocab_words)

    pd.DataFrame({'text': matched_ext}).to_csv("matched_text_data_without_stem_with_extended.csv", index=False, encoding='utf-8')
    pd.DataFrame({'text': unmatched_ext}).to_csv("unmatched_text_data_without_stem_with_extended.csv", index=False, encoding='utf-8')


if __name__ == "__main__":
    main()
