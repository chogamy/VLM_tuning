import os
import json
from typing import Any, ClassVar, List, Union, Dict

from PIL import Image
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from dataclasses import dataclass


def format_conversation(
    image: str | bytes | Image.Image,
    prefix: str,
    suffix: str | None = None,
    system_message: str | None = None,
) -> list[dict]:
    messages = []

    if system_message is not None:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prefix,
                },
            ],
        }
    )

    if suffix is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": suffix}],
            }
        )

    return messages


# I want to inherit DataCollatorMixin
# but can't import
@dataclass
class DataCollatorForImageTextToText:
    processor: Any
    system_message: str
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        images, data = zip(*examples)

        conversations = [
            format_conversation(
                image, entry["prefix"], entry["suffix"], self.system_message
            )
            for image, entry in zip(images, data)
        ]

        texts = [
            self.processor.apply_chat_template(
                conversation=conversation, tokenize=False
            )
            for conversation in conversations
        ]
        image_inputs = [
            process_vision_info(conversation)[0] for conversation in conversations
        ]
        model_inputs = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )

        labels = model_inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        image_tokens = [151652, 151653, 151655]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100

        for conversation_index, complete_conversation in enumerate(conversations):
            if len(complete_conversation) < 2:
                continue
            system_user_conversation = complete_conversation[:-1]
            system_user_text = self.processor.apply_chat_template(
                conversation=system_user_conversation, tokenize=False
            )
            system_user_image, _ = process_vision_info(system_user_conversation)
            system_user_model_inputs = self.processor(
                text=[system_user_text],
                images=[system_user_image],
                return_tensors="pt",
                padding=True,
            )
            system_user_input_length = system_user_model_inputs["input_ids"].shape[1]
            labels[conversation_index, :system_user_input_length] = -100

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = model_inputs["pixel_values"]
        image_grid_thw = model_inputs["image_grid_thw"]

        batch = {}
        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["pixel_values"] = pixel_values
        batch["image_grid_thw"] = image_grid_thw
        batch["labels"] = labels

        return batch


class JSONLDataset(Dataset):
    """
    A dataset for loading images and annotations from a JSON Lines (JSONL) file.

    This class reads annotation entries from a specified JSONL file and ensures that each entry
    contains the required keys and that the corresponding image file exists in the given directory.
    Entries that fail validation (due to JSON parsing errors, missing keys, or non-existent image files)
    are skipped with an appropriate warning logged.

    Parameters:
        annotations_path (str): Filesystem path to the JSONL file containing dataset annotations.
        images_directory_path (str): Filesystem path to the directory containing image files.

    Example:
        ```
        from roboflow import download_dataset, login
        from maestro.trainer.common.datasets.jsonl import JSONLDataset

        login()

        dataset = download_dataset("universe.roboflow.com/roboflow-jvuqo/pallet-load-manifest-json/2", "jsonl")
        ds = JSONLDataset(
            annotations_path=f"{dataset.location}/test/annotations.jsonl",
            images_directory_path=f"{dataset.location}/test"
        )
        len(ds)
        # 10
        ```
    """

    JSONL_FILENAME: ClassVar[str] = "annotations.jsonl"
    REQUIRED_KEYS: ClassVar[set[str]] = {"image", "prefix", "suffix"}

    def __init__(self, annotations_path: str, images_directory_path: str) -> None:
        super().__init__()
        self.image_directory_path = images_directory_path
        self.entries = self._load_entries(annotations_path, images_directory_path)

    @classmethod
    def _load_entries(cls, annotations_path: str, images_dir: str) -> list[dict]:
        """
        Load and validate dataset entries from a JSON Lines (JSONL) file.

        Reads each line in the specified file, attempts to parse it as JSON, and verifies that
        every resulting entry contains the required keys. Additionally, it ensures that the
        associated image file exists in the given directory. Entries that cannot be parsed or do not
        meet the validation criteria are skipped with a warning.

        Parameters:
            annotations_path (str): Filesystem path to the JSONL file.
            images_dir (str): Filesystem path to the image directory.

        Returns:
            list[dict]: A list of valid annotation dictionaries.
        """

        entries = []
        total_lines = 0
        skipped_count = 0

        with open(annotations_path, encoding="utf-8") as file:
            for line_idx, line in enumerate(file, start=1):
                total_lines += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    skipped_count += 1
                    continue
                missing_keys = cls.REQUIRED_KEYS - entry.keys()
                if missing_keys:
                    skipped_count += 1

                    continue
                image_path = os.path.join(images_dir, entry["image"])
                if not os.path.exists(image_path):
                    skipped_count += 1

                    continue
                entries.append(entry)

        return entries

    def __len__(self) -> int:
        """
        Return the number of valid entries in the dataset.

        Returns:
            int: Total count of dataset entries.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Image.Image, dict[str, Any]]:
        """
        Retrieve the image and its corresponding annotation entry at the specified index.

        Parameters:
            idx (int): The zero-based index of the desired entry.

        Returns:
            tuple: A tuple containing:
                - PIL.Image.Image: The image object.
                - dict: The corresponding annotation entry.

        Raises:
            IndexError: If the index is out of the valid range.
        """
        if idx >= len(self.entries):
            raise IndexError(f"Index {idx} is out of range.")
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry["image"])
        image = Image.open(image_path).convert("RGB")
        return image, entry
