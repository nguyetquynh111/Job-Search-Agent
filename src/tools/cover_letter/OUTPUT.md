# Output

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Job used for the letter. |
| `output_tex_path` | string | Generated LaTeX file. |
| `output_pdf_path` | string | Generated PDF file. |
| `page_count` | integer | PDF page count. |
| `evidence_used` | list of strings | Evidence IDs used in the letter. |
| `errors` | list of strings | Generation or compilation problems. |

Every evidence ID must exist in the input. Successful output paths must point to generated files.

