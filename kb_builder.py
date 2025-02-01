import os
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime

from google.colab import drive
from dotenv import load_dotenv
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import DoclingLoader, ExportType
from langchain_docling import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Milvus
from docling.chunking import HybridChunker