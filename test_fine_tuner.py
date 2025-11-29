
import unittest
import os
import pandas as pd
from fine_tuner import run_fine_tuning

class TestFineTuner(unittest.TestCase):

    def test_empty_dataset(self):
        """
        Tests that run_fine_tuning raises a ValueError when the dataset is empty.
        """
        # Create an empty CSV file
        empty_csv_path = "empty_dataset.csv"
        pd.DataFrame({'text': []}).to_csv(empty_csv_path, index=False)

        with self.assertRaises(ValueError) as context:
            run_fine_tuning(empty_csv_path)

        self.assertTrue("The uploaded dataset is empty." in str(context.exception))

        # Clean up the empty file
        os.remove(empty_csv_path)

    def test_no_text_column(self):
        """
        Tests that run_fine_tuning raises a ValueError when the dataset has no 'text' column.
        """
        # Create a CSV file without a 'text' column
        no_text_csv_path = "no_text_dataset.csv"
        pd.DataFrame({'data': ['some data']}).to_csv(no_text_csv_path, index=False)

        with self.assertRaises(ValueError) as context:
            run_fine_tuning(no_text_csv_path)

        self.assertTrue("Dataset must contain a 'text' column." in str(context.exception))

        # Clean up the file
        os.remove(no_text_csv_path)

if __name__ == '__main__':
    unittest.main()
