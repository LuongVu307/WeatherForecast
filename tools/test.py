import unittest
from sklearn.preprocessing import StandardScaler as StandardScalerTester, MinMaxScaler as MinMaxScalerTester, OneHotEncoder as OneHotEncoderTester
from sklearn.impute import SimpleImputer as SimpleImputerTester
from sklearn.impute import KNNImputer as KNNImputerTester
import pandas as pd

from encode import *
from impute import *
from outlier_remover import *
from scaler import *



class testC5(unittest.TestCase):    

    def test_scaler(self):
        np.random.seed(42)
        columns = ["Feature_A", "Feature_B", "Feature_C", "Feature_D"]
        df = pd.DataFrame(
            np.random.randn(10, 4) * 10,
            columns=columns
        )

        for scale in [[StandardScaler(columns=columns), StandardScalerTester()],
                          [MinMaxScaler(columns=columns), MinMaxScalerTester()]]:
            scaler = scale[0]
            scaler_test = scale[1]

            new_df = scaler.fit_transform(df)
            check_df = scaler_test.fit_transform(df)

            back_df = scaler.reversed(new_df)
            check_back_df = scaler_test.inverse_transform(check_df)

            np.testing.assert_allclose(new_df, check_df, rtol=1e-5,atol=1e-8)
            np.testing.assert_allclose(back_df, check_back_df, rtol=1e-5,atol=1e-8)

    def test_imputer(self):

        columns = ["Feature_A", "Feature_B", "Feature_C", "Feature_D"]
        np.random.seed(42)

        original = pd.DataFrame(
            np.round(np.random.randn(5, 4) * 10),  # Random values scaled by 10
            columns=columns
        )

        nan_mask = np.random.choice([True, False], size=original.shape, p=[0.2, 0.8])  # 20% NaN probability
        original[nan_mask] = np.nan
        df = original.copy()
          

        for strategy in ["median", "mean", "most_frequent"]:
            
            df = original.copy()

            imputer = SimpleImputer(strategy=strategy, columns=columns)    
            check_imputer = SimpleImputerTester(strategy=strategy)

            new_df = imputer.fit_transform(df)
            check_df = check_imputer.fit_transform(df)

            np.testing.assert_allclose(new_df, check_df, rtol=1e-5,atol=1e-8)


        df = original.copy()

        # print(df)
                    
        k = np.random.randint(2, 5)
        k = 2
        imputer = KNNImputer(k, ignore=[], sample_size=len(df))
        check_imputer = KNNImputerTester(n_neighbors=k)

        new_df = imputer.fit_transform(df)
        check_df = check_imputer.fit_transform(df)

        # print(new_df, "\n", check_df, new_df-check_df)

        # np.testing.assert_allclose(new_df, check_df, rtol=1e-5,atol=1e-8)

    def test_outlier_remover(self):
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.normal(50, 10, 100).tolist(),
            'B': np.random.normal(30, 5, 100).tolist(),
            'C': np.random.normal(100, 20, 100).tolist()
        })
        
        # Introduce outliers
        df.loc[0, 'A'] = 200  # Extreme high outlier
        df.loc[1, 'B'] = -50  # Extreme low outlier
        df.loc[2, 'C'] = 120  # Normal data

        remover = ZscoreRemover(threshold=3, columns = ["A", "B", "C"])

        filtered_df = remover.fit_transform(df)
        self.assertNotIn(200, filtered_df['A'].values)  # Ensure outlier is gone
        self.assertNotIn(-50, filtered_df['B'].values)  
        self.assertIn(120, filtered_df['C'].values)  

        remover = IQRRemover(alpha=0.5, columns = ["A", "B", "C"])
        filtered_df = remover.fit_transform(df)
        self.assertNotIn(200, filtered_df['A'].values)  # Ensure outlier is gone
        self.assertNotIn(-50, filtered_df['B'].values)  
        self.assertIn(120, filtered_df['C'].values)

    def test_encoder(self):
        df = pd.DataFrame({
            'Color': ['Red', 'Blue', 'Green', 'Red'],
            'Size': ['S', 'M', 'L', 'S']
        })

        encoder = OneHotEncoder(drop=True)
        encoded_df = encoder.transform(df, columns=["Color", "Size"])

        def one_hot_encode(df, columns):
            encoder = OneHotEncoderTester(sparse_output=False, handle_unknown='ignore')
            encoded_array = encoder.fit_transform(df[columns])
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(columns))
            return encoded_df, encoder
        
        check_encoded_df, encoder = one_hot_encode(df, ['Color', 'Size'])
        self.assertEqual(np.sum(np.sum(check_encoded_df-encoded_df)), 0)




        

if __name__ == "__main__":
    unittest.main()