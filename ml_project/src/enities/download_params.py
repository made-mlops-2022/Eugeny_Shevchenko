from dataclasses import dataclass


@dataclass()
class DownloadParams:
    s3_path: str
    output_path: str
    raw_filename: str
