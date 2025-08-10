# LLM Processor

This module is prepared to process validated interviewee sentences through an LLM API once direct API access is available.

## Status

🔄 **Waiting for Direct API Access**

We're currently waiting to get direct API access to the LLM service. Once available, this module will:

- Process validated interviewee sentences from our pipeline
- Submit them to the LLM API with the category "Software for Architects, General Contractors and Interior Designers"
- Save responses to `data/processed/llm_processed_interviewee/`

## Planned Features

- **🔌 Direct API Integration**: Connect directly to the LLM API
- **📁 Batch Processing**: Process multiple CSV files automatically
- **🔧 Chunking**: Handle large files by splitting into manageable chunks
- **📊 Organized Output**: Save results in structured directories
- **🔄 Retry Logic**: Handle network issues and retries failed submissions

## Integration with Pipeline

This processor will integrate with the existing pipeline:

1. **Disfluency Processing**: `process_interview_directory.py`
2. **CSV Generation**: Creates validated sentences
3. **LLM Processing**: This module (once API access is available)
4. **Results**: Saved to `llm_processed_interviewee` directory

## Next Steps

Once direct API access is obtained:

1. Update the processor with the actual API endpoints
2. Add authentication handling
3. Implement the processing logic
4. Test with our validated interviewee sentences

## Dependencies

- `requests`: HTTP requests
- `pandas`: CSV processing
- `tqdm`: Progress bars
- `pathlib2`: Path handling
