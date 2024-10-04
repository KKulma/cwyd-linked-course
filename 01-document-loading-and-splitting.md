### Langchain libraries

- langchain - the core langchain API
- langchain-openai - langchain's integration to OpenAI
- pypdf - loads pdf docs into an array of documents
- faiss-cpu - vector db
- langchainhub - helps sharing and discovering high-quality prompts


### Document loading
- see code snippets using pypdf



### Chunking strategies

- token count - splitting the text into chunks that contains a certain number of tokens 
- semantic completness - ensuring that chunks maintain semantic integrity
- contextual relevance - chunks are created in a way that retains the context necessary for understanding and processing

Check the list of the available langchain's text splitters [here](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/) 