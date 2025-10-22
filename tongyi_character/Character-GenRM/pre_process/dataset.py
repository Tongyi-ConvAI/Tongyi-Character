from datasets import load_dataset, DatasetDict
from abc import ABC, abstractmethod
import pandas as pd
from datasets import Dataset as HfDataset
import ast
import os
from pathlib import Path


# ---- Path utility: locate project root and data directory ----
_THIS_FILE = Path(__file__).resolve()
ROOT_DIR = _THIS_FILE.parent.parent          # P-GenRM/
DATA_DIR = ROOT_DIR               

def load_dataset_simple(path: str, *args, **kwargs):
    """
    Load a dataset from a local relative path under the project /data folder.
    Prevents accidentally downloading from Hugging Face Hub.
    """
    # resolve relative path to absolute local path
    abs_path = Path(path)
    if not abs_path.is_absolute():
        abs_path = (DATA_DIR / Path(path)).resolve()

    if not abs_path.exists():
        raise FileNotFoundError(f"Local dataset path not found: {abs_path}")

    print(f"Loading dataset from local path: {abs_path}")
    ds = load_dataset(str(abs_path), *args, **kwargs)
    print(f"Successfully loaded dataset from local path: {abs_path}")
    return ds



class Dataset(ABC):

    def __init__(self) -> None:
        super().__init__()

    def get_dataset(self):
        return self.ds

    def get_dataframe(self, split: str = "train"):
        if split == "all":
            return [self.ds[sp].to_pandas() for sp in self.ds.keys()]
        return self.ds[split].to_pandas()

    @abstractmethod
    def get_user_data_df(self, user_id: str):
        pass

    @abstractmethod
    def get_user_data(self, user_id: str):
        pass

    @abstractmethod
    def get_all_user_data(self):
        pass

    @abstractmethod
    def get_user_ids(self):
        pass


class SynthesizeMeDataset(Dataset):
    def __init__(self, hf_dataset_path: str = os.path.join("data",  "chatbot_arena_personalized_0125")):
        # Load dataset directly from relative path under data/
        self.ds = load_dataset_simple(hf_dataset_path)

        columns_to_remove = [
            'dataset', 'query_reasoning', 'personalizable_query', 'response_reasoning',
            'personalizable_responses', 'reasoning_gemini/gemini-1.5-pro',
            'prediction_flip_gemini/gemini-1.5-pro', 'reasoning_flip_gemini/gemini-1.5-pro',
            'reasoning_meta-llama/Llama-3.3-70B-Instruct', 'prediction_flip_meta-llama/Llama-3.3-70B-Instruct',
            'reasoning_flip_meta-llama/Llama-3.3-70B-Instruct', 'prediction_gemini/gemini-1.5-pro',
            'prediction_meta-llama/Llama-3.3-70B-Instruct', 'prediction_azure/gpt-4o-mini-240718',
            'reasoning_azure/gpt-4o-mini-240718', 'prediction_flip_azure/gpt-4o-mini-240718',
            'reasoning_flip_azure/gpt-4o-mini-240718', 'prediction_Qwen/Qwen2.5-72B-Instruct',
            'reasoning_Qwen/Qwen2.5-72B-Instruct', 'prediction_flip_Qwen/Qwen2.5-72B-Instruct',
            'reasoning_flip_Qwen/Qwen2.5-72B-Instruct', 'prediction_meta-llama/Llama-3.1-70B-Instruct',
            'reasoning_meta-llama/Llama-3.1-70B-Instruct', 'prediction_flip_meta-llama/Llama-3.1-70B-Instruct',
            'reasoning_flip_meta-llama/Llama-3.1-70B-Instruct', 'agreement'
        ]

        # Remove only columns that actually exist
        existing_columns = set(self.ds.column_names)
        to_remove = [col for col in columns_to_remove if col in existing_columns]
        if to_remove:
            self.ds = self.ds.remove_columns(to_remove)

        self.joint_df = pd.concat([self.ds[split].to_pandas() for split in self.ds.keys()])

    def get_dataframe(self, split: str = "train"):
        if split == "all":
            return self.joint_df
        return self.ds[split].to_pandas()

    def get_user_data_df(self, user_id: str):
        return self.joint_df[self.joint_df['user_id'] == user_id]

    def get_user_data(self, user_id: str):
        df = self.get_user_data_df(user_id)
        train_data = df[df['split'] == 'train'].to_dict(orient='records')
        val_data = df[df['split'] == 'val'].to_dict(orient='records')
        test_data = df[df['split'] == 'test'].to_dict(orient='records')
        return train_data, val_data, test_data

    def get_all_user_data(self, split: str = "all"):
        df = self.get_dataframe(split)
        grouped = df.groupby('user_id')
        return {
            user_id: (
                group[group['split'] == 'train'].to_dict(orient='records'),
                group[group['split'] == 'val'].to_dict(orient='records'),
                group[group['split'] == 'test'].to_dict(orient='records')
            )
            for user_id, group in grouped
        }

    def get_train_val_user_data(self, split: str = "all"):
        df = self.get_dataframe(split)
        grouped = df.groupby('user_id')
        return {
            user_id: (
                group[group['split'] == 'train'].to_dict(orient='records'),
                group[group['split'] == 'val'].to_dict(orient='records')
            )
            for user_id, group in grouped
        }

    def get_user_ids(self, split: str = "all"):
        df = self.get_dataframe(split)
        return df['user_id'].unique()

class ChatbotArenaDataset(SynthesizeMeDataset):
    def __init__(self):
        super().__init__(hf_dataset_path=os.path.join("data",  "chatbot_arena_personalized_0125"))



class PrismDataset(SynthesizeMeDataset):
    def __init__(self):
        super().__init__(hf_dataset_path=os.path.join("data", "prism_personalized_0125"))


class Prism_personal_align_Dataset(SynthesizeMeDataset):
    def __init__(self):
        # 1) Load main dataset: prism_personalized_0125
        super().__init__(hf_dataset_path=os.path.join("data", "prism_personalized_0125"))

        # 2) Load survey data (user profiles)
        survey_ds = load_dataset_simple(
            os.path.join("data",  "prism-alignment"), name="survey"
            
        )
        survey_df = survey_ds['train'].to_pandas()
        profile_columns = [
            'user_id', 'self_description', 'system_string', 'age', 'gender',
            'employment_status', 'education', 'marital_status', 'study_locale',
            'religion', 'ethnicity', 'stated_prefs'
        ]
        survey_profile_df = survey_df[profile_columns]

        # 3) Merge profile info into main dataset
        merged_df = pd.merge(self.joint_df, survey_profile_df, on='user_id', how='left')

        # 4) Load conversations (choice_attributes)
        conversations_ds = load_dataset_simple(
            os.path.join("data",  "prism-alignment"), name="conversations"
        )
        conversations_df = conversations_ds['train'].to_pandas()
        conv_attributes_df = conversations_df[['conversation_id', 'choice_attributes']]

        # 5) Merge choice_attributes into dataset
        merged_df_with_attrs = pd.merge(merged_df, conv_attributes_df, on='conversation_id', how='left')
        print(f"Total rows after merging choice_attributes: {len(merged_df_with_attrs)}")

        # 6) Format choice_attributes dictionaries as readable strings
        def format_choice_attributes(attr_dict):
            if not isinstance(attr_dict, dict):
                return ""
            return "\n".join([f"{key}: {value}" for key, value in attr_dict.items()])

        new_column_name = 'formatted_choice_attributes'
        merged_df_with_attrs[new_column_name] = merged_df_with_attrs['choice_attributes'].apply(format_choice_attributes)
        merged_df_with_attrs.drop(columns=['choice_attributes'], inplace=True)

        # 7) Keep only train/val/test splits
        final_df = merged_df_with_attrs[merged_df_with_attrs['split'].isin(['train', 'val', 'test'])].copy()
        self.joint_df = final_df


if __name__ == "__main__":
    ds = Prism_personal_align_Dataset()
    ids = ds.get_user_ids()
    train, val, test = ds.get_all_user_data()[ids[13]]
    print("Example test data for user:", ids[13])
    print(test)
