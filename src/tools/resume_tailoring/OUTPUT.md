# Output

| Field | Type | Meaning |
| --- | --- | --- |
| `job_id` | string | Job used for tailoring. |
| `status` | string | Result status. |
| `output_tex_path` | string | Generated LaTeX file. |
| `output_pdf_path` | string | Generated PDF file. |
| `page_count` | integer | PDF page count. |
| `change_log` | list of changes | What changed and which evidence supports it. |
| `errors` | list of strings | Generation or compilation problems. |

Successful output paths must point to new files. The source resume must never be overwritten.

