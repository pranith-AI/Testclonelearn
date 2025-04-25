'''
-----------------------------------------------------------------------
File: __init__.py
Creation Time: Feb 8th 2024, 2:59 pm
Author: Saurabh Zinjad
Developer Email: saurabhzinjad@gmail.com
Copyright (c) 2023-2024 Saurabh Zinjad. All rights reserved | https://github.com/Ztrimus
-----------------------------------------------------------------------
'''
"""ZLM (Zinjad Language Model) Package."""

import os
import json
import re
import validators
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from zlm.schemas.sections_schemas import ResumeSchema
from zlm.utils import utils
from zlm.utils.latex_ops import latex_to_pdf
from zlm.utils.llm_models import ChatGPT, Gemini, OllamaModel
from zlm.utils.data_extraction import read_data_from_url, extract_text
from zlm.utils.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity, vector_embedding_similarity
from zlm.prompts.resume_prompt import CV_GENERATOR, RESUME_WRITER_PERSONA, JOB_DETAILS_EXTRACTOR, RESUME_DETAILS_EXTRACTOR
from zlm.schemas.job_details_schema import JobDetails
from zlm.variables import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, LLM_MAPPING, section_mapping

module_dir = os.path.dirname(__file__)
demo_data_path = os.path.join(module_dir, "demo_data", "user_profile.json")
prompt_path = os.path.join(module_dir, "prompts")


class AutoApplyModel:
    """Class for automatically generating resumes and CVs using LLM models."""
    
    def __init__(
        self, 
        api_key: str, 
        provider: str = "openai",
        model: str = "gpt-3.5-turbo-16k",
        downloads_dir: Optional[str] = None
    ):
        """Initialize AutoApplyModel.
        
        Args:
            api_key: API key for the LLM provider
            provider: Name of LLM provider ('openai' or 'gemini')
            model: Name of the model to use
            downloads_dir: Directory to save generated files
        """
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model
        self.downloads_dir = downloads_dir or "output"
    
    def resume_cv_pipeline(self, url: str, master_data: Dict[str, Any]) -> None:
        """Generate resume/CV based on job posting and master data.
        
        Args:
            url: URL of the job posting
            master_data: Dictionary containing user's master data
        """
        # Validate URL
        if not validators.url(url):
            raise ValueError("Invalid URL provided")
            
        # Create downloads directory if it doesn't exist
        if not os.path.exists(self.downloads_dir):
            os.makedirs(self.downloads_dir)
            
        # Load master data if string path provided
        if isinstance(master_data, str):
            with open(master_data) as f:
                master_data = json.load(f)
                
        # TODO: Implement resume generation logic
        pass
