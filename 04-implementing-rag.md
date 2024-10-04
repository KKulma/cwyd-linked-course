### Basic RAG chains

Chains are a fundamental concept of LangChain.

A chain is a sequence of calls connected together where the output of one call becomes the input to the next call. 

These chains can include calls to an LLM, API, a custom functions, call to vector database etc.

We'll start with a standard prompt template

A prompt template - a reusable template usually shared across organizations that standardizes the way people prompt LLMs

Our LangChain chain:
- take user's queries 
- searches manuscripts to look for the answer to the question
- passes the retrieved context and an initial question to the LLM
- returns the answer


Retriever helps us build a retrival chain which pulls data from our vector database and passes that into the prompt template

```
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
```

The basic chain (without conversation history) looks like this: 

```{python}
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#combine multiple steps in a single chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser() #convert the chat message to a string
)
```


### How to preserve conversation history

To preserve conversation history, we don't use the prompt template but this instead:

```
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """Given the chat history and a recent user question \
generate a new standalone question \
that can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed or otherwise return it as is."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever_with_history = create_history_aware_retriever(
    llm, retriever, prompt
)
```

Notice the retriever - it's a chain that takes the conversation history and returns documents


### Question answering 

Let's tie everything together to perform question answering. We'll send the input query, along with the chat history, as the prompt to the LLM. We will contextualize our prompt and use a retriever to pull data from the vector database and update the prompt before it is sent to the LLM. Then, we will retrieve the answer from the LLM. 

`create_stuff_document_chain()` -- builds the full Q&A chain using the create_stuff_documents_chain function. This function takes a list of documents, formats them all into a prompt, and passes that prompt to an LLM. We'll use this function to generate the question-answer chain in this variable that accepts the retrieved context alongside the conversation history and query to generate the answer. This approach will pass all documents, so we'll need to ensure that it fits within the context window of our LLM.
