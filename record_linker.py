import dedupe
import os
import pandas as pd
import numpy as np
import json


def find_matches(LHS: pd.DataFrame, RHS: pd.DataFrame):
    # Setup

    settings_file = 'record_linker_settings\\name_matching_learned_settings'
    training_file = 'record_linker_settings\\name_matching_training.json'

    LHS = LHS.to_dict('index')
    RHS = RHS.to_dict('index')

    # print('Missing values in {}'.format(pd.DataFrame(left_file)))
    # data_1_cols_with_missing = missing_perc(data_1)
    # print('\nMissing values in {}'.format(pd.DataFrame(right_file)))
    # data_2_cols_with_missing = missing_perc(data_2)

    # ## Training

    if os.path.exists(settings_file):
        print('reading from', settings_file)
        with open(settings_file, 'rb') as sf:
            linker = dedupe.StaticRecordLink(sf)

    else:
        # Define the fields the linker will pay attention to
        #
        # Notice how we are telling the linker to use a custom field comparator
        # for the 'price' field.
        fields = [
            {'field': 'name', 'type': 'String', 'has missing': False},
        ]

        # Create a new linker object and pass our data model to it.
        linker = dedupe.RecordLink(fields)

        # If we have training data saved from a previous run of linker,
        # look for it an load it in.
        # __Note:__ if you want to train from scratch, delete the training_file
        if os.path.exists(training_file):
            print('reading labeled examples from ', training_file)
            with open(training_file) as tf:
                linker.prepare_training(LHS,
                                        RHS,
                                        training_file=tf) # sample size 150.000
        else:
            print('prepare for training...')
            linker.prepare_training(LHS, RHS)

            # ## Active learning
            # Dedupe will find the next pair of records
            # it is least certain about and ask you to label them as matches
            # or not.
            # use 'y', 'n' and 'u' keys to flag duplicates
            # press 'f' when you are finished
            print('starting active labeling...')

            dedupe.console_label(linker)

            linker.train()

            # When finished, save our training away to disk
            with open(training_file, 'w') as tf:
                linker.write_training(tf)

            # Save our weights and predicates to disk.  If the settings file
            # exists, we will skip all the training and learning next time we run
            # this file.
            with open(settings_file, 'wb') as sf:
                linker.write_settings(sf)

        # ## Blocking

        # ## Clustering

        # Find the threshold that will maximize a weighted average of our
        # precision and recall.  When we set the recall weight to 2, we are
        # saying we care twice as much about recall as we do precision.
        #
        # If we had more data, we would not pass in all the blocked data into
        # this function but a representative sample.

    linked_records = linker.join(LHS, RHS, 0.5, 'many-to-many')

    return pd.DataFrame(linked_records)


if __name__ == '__main__':

    ### Initialize LHS
    path_lhs = r'C:\Users\Oleg\Documents\Masterarbeit\STT\imdb_speechbrain_10000_last_names_robust\validation_dataset.csv'
    val_df = pd.read_csv(path_lhs)

    # create a list of misspelling and an according list of labels
    samples = list(val_df[['mistake_1', 'mistake_2', 'mistake_3', 'mistake_4']].values.flatten())
    sample_labels = np.repeat(val_df['label_10000'], 4)

    LHS = {name: label for name, label in zip(samples, sample_labels)}

    ### Initialize RHS

    path_rhs = r'C:\Users\Oleg\Documents\Masterarbeit\STT\imdb_speechbrain_10000_last_names_robust\labels_dict.json'

    with open(path_rhs, 'r', encoding='utf-8') as f:
        rhs = json.loads(f.read())

    RHS = {name: label for label, name in rhs.items()}

    LHS = pd.DataFrame(zip(*[LHS.keys(), list(map(str, LHS.values()))]), columns=['name', 'label'])
    RHS = pd.DataFrame(zip(*[RHS.keys(), list(map(str, RHS.values()))]), columns=['name', 'label'])

    match = find_matches(LHS, RHS)
    match.to_csv('dedupe_matches.csv')
