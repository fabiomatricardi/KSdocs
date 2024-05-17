###############################################################
# Version with higher resolution 
# splitting for the QnA section
# Keeping it to 1200 tokens gives too little
# informations about the text
################################################################

#GENERAL CALL IMPORT
from tqdm.rich import trange, tqdm
from rich.markdown import Markdown
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from rich.console import Console
console = Console(width=90)
from llama_cpp import Llama
import tiktoken
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter
encoding = tiktoken.get_encoding("r50k_base")

#from sentence_transformers import CrossEncoder

## Logger file
selectedmodel = 'h2o-danube-1.8b-chat.Q5_K_M.gguf'
tstamp = datetime.datetime.now()
tstamp = str(tstamp).replace(' ','_')
tstamp = str(tstamp).replace(':','_')
logfile = f'{selectedmodel}_{tstamp[:-7]}.txt'
dbfilename = f'{tstamp[:-7]}_db.pkl'



console.clear()
console.print(f'[green1 bold]0. Loading LLm Models...')
console.print(f'[grey78]1. Loading Document to Analyze...')
console.print(f'[grey78]2. Generating main info...')
console.print(f'[grey78]3. Generating General QnA by Sections...')
console.print(f'[grey78]4. Generating List of Main Document Topics...')
console.print(f'[grey78]5. Generating Document Table of Contents...')
console.print(f'[grey78]6. Generating Document SUMMARY...')
console.print(f'[grey78]7. Saving the QnA database into picklefile: {dbfilename}')
# Load the model, here we use our base sized model
# console.print('[red bold]Loading Cross Encoder')
# model = CrossEncoder('encoder')  # 17 MegaByte !! "cross-encoder/ms-marco-TinyBERT-L-2"
selectedmodel = 'stablelm-zephyr-3b.Q5_K_M.gguf'
console.print('Loading new model')
mp = 'model\stablelm-zephyr-3b.Q4_K_M.gguf'
from llama_cpp import Llama
console.print("Loading ✅✅✅✅ stablelm-zephyr-3b.Q4_K_M.gguf with LLAMA.CPP...")
llm = Llama(model_path=mp,n_ctx=4096, n_gpu_layers=-1,verbose=False)

######## FUNCTIONS

def writehistory(filename,text):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()
#Write in the history the first 2 sessions
writehistory(logfile,f'Your own DOCUMENT PIPELINE with ✅✅✅✅ {selectedmodel} and LLAMA.CPP\n---\n')   

#Load PDF Function
import os
import fitz #pyMuPDF
#miofile = "/content/28884E00- SYSTEM OPERATIONAL TEST PROCEDURE PREPARATION CUIDELINE.pdf"
def LoadPDFandWork(filepath,chunks, overlap):
  """
  pass a file path, int chunk and overlap
  return a list d of text chunks and full article text
  """
  from langchain_community.document_loaders import TextLoader
  from langchain.text_splitter import TokenTextSplitter
  TOKENtext_splitter = TokenTextSplitter(chunk_size=chunks, chunk_overlap=overlap)
  #splitted_text = TOKENtext_splitter.split_text(fulltext) #create a list
  from langchain_community.document_loaders import PyMuPDFLoader
  import datetime
  start = datetime.datetime.now()
  console.print('1. loading pdf')
  loader = PyMuPDFLoader(filepath) #on Win local simply 'stl-0000011.pdf'
  data = loader.load_and_split(TOKENtext_splitter)
  delta = datetime.datetime.now() - start
  console.print(f'2. Loaded in {delta}')
  console.print(f'3. Number of items: {len(data)}')
  console.print('---')
  its = 0
  chars = 0
  solotesto = ''
  for items in data:
      testo = len(items.page_content)
      solotesto = solotesto + ' ' + items.page_content
      #console.print(f"Number of CHAR in Document {its}: {testo}")
      its = its + 1
      chars += testo

  console.print('---')
  console.print(f'> Total lenght of text in characthers: {chars}')
  console.print('---')
  context_count = len(encoding.encode(solotesto))
  console.print(f"Number of Tokens in the Article: {context_count}")
  d = []
  for items in data:
    d.append(items.page_content)
  return d,solotesto
"""
d,article =  LoadPDFandWork(miofile, 300,50)
"""

#lOAD A TXT FILE
#FOR TXT
# filename = '/content/2024-04-11 12.52.28 Kaggle s wrong turn when AI becomes a teacher and.txt'
def LoadTXT(filename, chunks, overlap):
    """
    pass a file path, int chunk and overlap
    return a list d of text chunks and full article text
    """
    with open(filename, encoding='utf-8') as f:
        article = f.read()
    f.close()
    import tiktoken
    encoding = tiktoken.get_encoding("r50k_base")
    context_count = len(encoding.encode(article))
    console.print(f"Number of Tokens in the Article: {context_count}")  
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import TokenTextSplitter
    TOKENtext_splitter = TokenTextSplitter(chunk_size=chunks, chunk_overlap=overlap)
    d = TOKENtext_splitter.split_text(article) #create a list
    console.print(f"Number of Document Chunks in the Article: {len(d)}") 
    return d, article
"""
d,article =  LoadTXT(miofile, 1200,50)
"""



## Load a llama-cpp-python quantized model
"""
mp = 'model\stablelm-zephyr-3b.Q4_K_M.gguf'
from llama_cpp import Llama
console.print("Loading ✅✅✅✅ stablelm-zephyr-3b.Q5_K_S.gguf with LLAMA.CPP...")
llm = Llama(model_path=mp,n_ctx=4096, n_gpu_layers=-1,verbose=False)
"""
"""
mp = 'model\h2o-danube-1.8b-chat.Q5_K_M.gguf'
from llama_cpp import Llama
console.print(f"Loading ✅✅✅✅ {selectedmodel} with LLAMA.CPP...")
llm = Llama(model_path=mp,n_ctx=4096, n_gpu_layers=-1,verbose=False)
""" 


def RaG(llamaCPP,context, question):
  prompt = f'''Read this and answer the question.
[context]
{context}
[end of context]
Answer the questions only using the given information in the context provided.
If there's no clear answer in the context, reply 'unanswerable' and nothing else.
In cases of uncertainty due to insufficient information, always mark it as 'unanswerable' and nothing else
Avoid assuming or guessing information not explicitly stated in the context provided.
Your aim is to find accurate answers solely from the context, without making assumptions or speculations.
Evaluate the information carefully and respond accordingly, remember to say 'unanswerable' if no answer can be found in context.

Question: {question}'''
  messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "user", "content": prompt}
    ]
  start = datetime.datetime.now()
  result = llamaCPP.create_chat_completion(
                    messages=messages,
                    max_tokens=300,
                    stop=["</s>","[/INST]","/INST","<|endoftext|>"],
                    temperature = 0,
                    repeat_penalty = 1.2)
  delta = datetime.datetime.now() - start
  #console.print(f'[red1 bold]Question: {question}')
  #console.print(result["choices"][0]["message"]["content"])
  #console.print('---')
  #console.print(f'Generated in {delta}')
  return result["choices"][0]["message"]["content"]

def TableOfContents(llamaCPP,context,maxtokens):
  prompt = f'''Given a block of text, please extract and return the Table of Contents (ToC) or an outline in a 
formatted list format that represents the main headings and subheadings present within the text. Please ensure
that each heading is numbered sequentially and accurately reflects its position within the context of the 
overall content.
Note: The Table of Contents should not include any unnecessary or repetitive headings that are only present 
for formatting purposes, such as page numbers or section headers (e.g., "Section 1" vs. Page 23).

[start of context]
{context}
[end of context]

Table of Contents:
'''
  messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "user", "content": prompt}
    ]
  start = datetime.datetime.now()
  result = llamaCPP.create_chat_completion(
                    messages=messages,
                    max_tokens=maxtokens,
                    stop=["</s>","[/INST]","/INST","<|endoftext|>"],
                    temperature = 0,
                    repeat_penalty = 1.2)
  delta = datetime.datetime.now() - start
  #console.print(f'[red1 bold]Table Of Contents')
  #console.print(result["choices"][0]["message"]["content"])
  #console.print('---')
  #console.print(f'Generated in {delta}')
  return result["choices"][0]["message"]["content"]


def get_TOC(llamaCPP,article):
  TOKENtext_splitter = TokenTextSplitter(chunk_size=1200, chunk_overlap=50)
  d_TOC = TOKENtext_splitter.split_text(article) #create a list
  console.print(f"Number of Document Chunks in the Article for TABLE OF CONTENT Generation: {len(d_TOC)}") 
  stackstart = datetime.datetime.now()
  tempTOC =''
  i = 1
  steps = len(d_TOC)
  for items in d_TOC:
    console.print(f'executing step {i} of {steps}')
    toc = TableOfContents(llamaCPP,items,300)
    tempTOC = tempTOC + '\n' + toc
    i += 1
  console.print('[red bold]Generating final TOC')
  finalTOC = TableOfContents(llamaCPP,tempTOC,600)
  stackdelta = datetime.datetime.now() - stackstart
  #console.print('---')
  #console.print(f'[red bold]{finalTOC}')
  #console.print('---')
  #console.print(f'[purple bold]STACK Generated in {stackdelta}')  
  return finalTOC

# context = Text to summarize:
def sumitall(context,string,maxlenght):
  prompt = f"""Write a short summary of the given this text extracts:
[start of context]
{context}
[end of context]

Summary:"""
  messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "user", "content": prompt}
    ]
  start = datetime.datetime.now()
  result = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=maxlenght,
                    stop=["</s>","[/INST]","/INST","<|endoftext|>"],
                    temperature = 0.1,
                    repeat_penalty = 1.2)
  delta = datetime.datetime.now() - start
  #console.print(f'[red1 bold]{string}>>')
  #console.print(result["choices"][0]["message"]["content"])
  #console.print('---')
  #console.print(f'Generated in {delta}')
  return result["choices"][0]["message"]["content"]

def finalSUM(listOfDOC):
  partialSum = ''
  job = len(listOfDOC)
  for items in trange(job):
    partialSum += sumitall(listOfDOC[items],'PARTIAL SUMMARY',350) + '  '
  #console.print('---')
  finalsummary = sumitall(partialSum,'FInal SUMMARY',500)
  return finalsummary

def topics_3(llamaCPP,articlechunks):
  finalist = []
  question_chunks = articlechunks #create a list
  for context in question_chunks:
    prompt = f"""Given this text extracts:
-----
{context}
-----

write the three relevant topics into a list. do not add any other text, return only the list.

"""
    messages = [
          {"role": "system", "content": "You are a helpful assistant.",},
          {"role": "user", "content": prompt}
      ]
    start = datetime.datetime.now()
    #console.print("> Main 3 topics")
    result = llamaCPP.create_chat_completion(
                      messages=messages,
                      max_tokens=400,
                      stop=["</s>","[/INST]","/INST"],
                      temperature = 0,
                      repeat_penalty = 1.2)
    delta = datetime.datetime.now() - start
    #console.print('done')
    #console.print(f'Generated in {delta}')
    #console.print(result["choices"][0]["message"]["content"])
    listato = result["choices"][0]["message"]["content"].split('\n')
    for s in listato:
      finalist.append(s)
  #console.print('---')
  #console.print(topicslist)
  return finalist





########################### PROGRAM START #############################################################################
console.clear()
console.print(f'[grey78]0. Loading LLm Models...')
console.print(f'[green1 bold]1. Loading Document to Analyze...')
console.print(f'[grey78]2. Generating main info...')
console.print(f'[grey78]3. Generating General QnA by Sections...')
console.print(f'[grey78]4. Generating List of Main Document Topics...')
console.print(f'[grey78]5. Generating Document Table of Contents...')
console.print(f'[grey78]6. Generating Document SUMMARY...')
console.print(f'[grey78]7. Saving the QnA database into picklefile: {dbfilename}')
import easygui   #https://easygui.readthedocs.io/en/master/api.html
path = easygui.fileopenbox(filetypes = ["*.txt"])
console.print(f'Filename: {path}')
# '2024-04-12 13.16.08 Kaggle s wrong turn when AI becomes a teacher and.txt'

#### CHUNKS FOR THE TOC AND SUMMARY
d,article =  LoadTXT(path, 1200,50)

#### CHUNKS FOR THE QUESTION ANSWERING
d2,article2 =  LoadTXT(path, 550,50)

a = input('is it ok (Yes/No)? ')
import sys   #https://www.askpython.com/python/examples/exit-a-python-program
if a.lower() == 'no':
  sys.exit("You don't like it right?")
replies = []
mainstart = datetime.datetime.now()
writehistory(logfile,f'Loading  ✅✅✅✅ {path} DOCUMENT\n---\n') 
################################## GENERATE MAIN INFO ON DOCUMENT  ################################################



console.clear()
console.print(f'[grey78]0. Loading LLm Models...')
console.print(f'[grey78]1. Loading Document to Analyze...')
console.print(f'[green1 bold]2. Generating main info...')
console.print(f'[grey78]3. Generating General QnA by Sections...')
console.print(f'[grey78]4. Generating List of Main Document Topics...')
console.print(f'[grey78]5. Generating Document Table of Contents...')
console.print(f'[grey78]6. Generating Document SUMMARY...')
console.print(f'[grey78]7. Saving the QnA database into picklefile: {dbfilename}')

writehistory(logfile,f'Generating main info')
writehistory(logfile,f'-------------')
incipit = article[:600]
generalQ = [
    "What is the title of the article?",
    "The article is written by? Who is the author?",
    "what is the web address of the article?"
]

for item in generalQ:
  question = item
  answer = RaG(llm,incipit, question)
  writehistory(logfile,question)
  writehistory(logfile,answer)
  writehistory(logfile,'---')
  if 'unanswerable' in answer:
    console.print('😱no good')
  else:
    replies.append({
      'question' : question,
      'answer' : answer
  })

#console.print('---')
#console.print(replies)


############################# GENERATE QUESTIONS AND ANSWERS #######################################
def questions_3(llamaCPP,articlechunk):
  finalist = []
  context = articlechunk
  prompt = f"""Given the following context:
[start of context]
{context}
[end of context]

Write the three main questions into a list
return only the three questions related to the provided context.
"""
  #console.print(prompt)
  #console.print('------------------------------------------------------------------------')
  messages = [
        {"role": "system", "content": "You are a helpful assistant.",},
        {"role": "user", "content": prompt}
    ]
  start = datetime.datetime.now()
  #console.print("> Main 3 questions")
  result = llamaCPP.create_chat_completion(
                    messages=messages,
                    max_tokens=250,
                    stop=["</s>","[/INST]","/INST"],
                    temperature = 0.1,
                    repeat_penalty = 1.2)
  delta = datetime.datetime.now() - start
  #console.print(result["choices"][0]["message"]["content"])
  #console.print('done')
  #console.print(f'Generated in {delta}')
  #console.print(result["choices"][0]["message"]["content"])
  tempreply = result["choices"][0]["message"]["content"]
  order = tempreply.find('1. ')
  listato = tempreply[order:].split('\n')
  for s in listato:
    qst = f'- {s}'
    finalist.append(qst)
  #console.print('---')
  #console.print(topicslist)
  return finalist

console.clear()
console.print(f'[grey78]0. Loading LLm Models...')
console.print(f'[grey78]1. Loading Document to Analyze...')
console.print(f'[grey78]2. Generating main info...')
console.print(f'[green1 bold]3. Generating General QnA by Sections...')
console.print(f'[grey78]4. Generating List of Main Document Topics...')
console.print(f'[grey78]5. Generating Document Table of Contents...')
console.print(f'[grey78]6. Generating Document SUMMARY...')
console.print(f'[grey78]7. Saving the QnA database into picklefile: {dbfilename}')

suggestedquestions = ''
for items in d2:
  list_of_questions = questions_3(llm,items)
  #console.print(list_of_questions)
  #console.print('------------------------\nFIXED LIST\n')
  mainquestions = []
  for i in list_of_questions:
    mainquestions.append(i[2:])
    suggestedquestions = suggestedquestions + i[2:] + '\n'
  #console.print(mainquestions)
  #console.print('------------------')
  for q in mainquestions:
    answer = RaG(llm,items, q)
    writehistory(logfile,q)
    writehistory(logfile,answer)
    writehistory(logfile,'---')
    if 'unanswerable' in answer:
      console.print('😱no good')
    else:
      replies.append({
        'question' : q,
        'answer' : answer
    })    
replies.append({
      'question' : 'What are some suggested questions?',
      'answer' : suggestedquestions})
replies.append({
      'question' : 'What are the main questions from the text?',
      'answer' : suggestedquestions})
replies.append({
      'question' : 'Questions?',
      'answer' : suggestedquestions})
#######################################################################################################


########################## GENERATE LIST OF TOPICS ####################################################

console.clear()
console.print(f'[grey78]0. Loading LLm Models...')
console.print(f'[grey78]1. Loading Document to Analyze...')
console.print(f'[grey78]2. Generating main info...')
console.print(f'[grey78]3. Generating General QnA by Sections...')
console.print(f'[green1 bold]4. Generating List of Main Document Topics...')
console.print(f'[grey78]5. Generating Document Table of Contents...')
console.print(f'[grey78]6. Generating Document SUMMARY...')
console.print(f'[grey78]7. Saving the QnA database into picklefile: {dbfilename}')

"""
del llm
selectedmodel = 'h2o-danube-1.8b-chat.Q5_K_M.gguf'
writehistory(logfile,f'Loading  ✅✅✅✅ {selectedmodel} with LLAMA.CPP\n---\n')  
console.print('loading new model')
mp = 'model\h2o-danube-1.8b-chat.Q5_K_M.gguf'
from llama_cpp import Llama
console.print(f"Loading ✅✅✅✅ {selectedmodel} with LLAMA.CPP...")
llm = Llama(model_path=mp,n_ctx=4096, n_gpu_layers=-1,verbose=False)
"""

miotopic = topics_3(llm,d)
maintopics = ''
for items in miotopic:
  maintopics += '- '+items[2:]+'\n'
#console.print(maintopics.replace('.',''))
replies.append({
      'question' : 'What are the main topics?',
      'answer' : maintopics.replace('.','')})
replies.append({
      'question' : 'What are the topics?',
      'answer' : maintopics.replace('.','')})
replies.append({
      'question' : 'Main topics?',
      'answer' : maintopics.replace('.','')})
writehistory(logfile,f'MAIN TOPICS\n{maintopics}\n-------')
#console.print(f'MAIN TOPICS\n{maintopics}\n----------------')


########################## TABLE OF CONTENT  ####################################

console.clear()
console.print(f'[grey78]0. Loading LLm Models...')
console.print(f'[grey78]1. Loading Document to Analyze...')
console.print(f'[grey78]2. Generating main info...')
console.print(f'[grey78]3. Generating General QnA by Sections...')
console.print(f'[grey78]4. Generating List of Main Document Topics...')
console.print(f'[green1 bold]5. Generating Document Table of Contents...')
console.print(f'[grey78]6. Generating Document SUMMARY...')
console.print(f'[grey78]7. Saving the QnA database into picklefile: {dbfilename}')

del llm
selectedmodel = 'h2o-danube-1.8b-chat.Q5_K_M.gguf'
writehistory(logfile,f'Loading  ✅✅✅✅ {selectedmodel} with LLAMA.CPP\n---\n')  
console.print('loading new model')
mp = 'model\h2o-danube-1.8b-chat.Q5_K_M.gguf'
from llama_cpp import Llama
console.print(f"Loading ✅✅✅✅ {selectedmodel} with LLAMA.CPP...")
llm = Llama(model_path=mp,n_ctx=4096, n_gpu_layers=-1,verbose=False)

writehistory(logfile,f'Generating Table of Contents')
writehistory(logfile,f'-------------')
finalTOC = get_TOC(llm,article)
order = finalTOC.find('1.')
def_finalTOC = finalTOC[order:]
writehistory(logfile,def_finalTOC)
replies.append({
      'question' : 'What is the index?',
      'answer' : def_finalTOC})
replies.append({
      'question' : 'What is the Table of Contents?',
      'answer' : def_finalTOC})
replies.append({
      'question' : 'Table of Contents?',
      'answer' : def_finalTOC})

########################## GENERATE SUMMARY  ####################################

console.clear()
console.print(f'[grey78]0. Loading LLm Models...')
console.print(f'[grey78]1. Loading Document to Analyze...')
console.print(f'[grey78]2. Generating main info...')
console.print(f'[grey78]3. Generating General QnA by Sections...')
console.print(f'[grey78]4. Generating List of Main Document Topics...')
console.print(f'[grey78]5. Generating Document Table of Contents...')
console.print(f'[green1 bold]6. Generating Document SUMMARY...')
console.print(f'[grey78]7. Saving the QnA database into picklefile: {dbfilename}')


writehistory(logfile,f'Generating SUMMARY')
finalsUmmarY = finalSUM(d)
writehistory(logfile,finalsUmmarY)

replies.append({
    'question' : 'What is the summary?',
    'answer' : finalsUmmarY
})
replies.append({
    'question' : 'What is the abstract of the article?',
    'answer' : finalsUmmarY
})
replies.append({
    'question' : 'What is the article about?',
    'answer' : finalsUmmarY
})
replies.append({
    'question' : 'What is this about?',
    'answer' : finalsUmmarY
})
replies.append({
    'question' : 'What is this text about?',
    'answer' : finalsUmmarY
})

#### GNERAL CLOSURE
maindelta = datetime.datetime.now() - mainstart


def SaveFirstData(datafilename,datadocs):
  import pickle
  ## SAVE IN PICKLE THE DOCUMENTS SET WITH METADATA
  output = open(datafilename, 'wb')
  pickle.dump(datadocs, output)
  output.close()
  console.print(Markdown(f"> Documents Data saved in {datafilename}..."))
  console.print(" - ")


console.clear()

console.print(f'[grey78]0. Loading LLm Models...')
console.print(f'[grey78]1. Loading Document to Analyze...')
console.print(f'[grey78]2. Generating main info...')
console.print(f'[grey78]3. Generating General QnA by Sections...')
console.print(f'[grey78]4. Generating List of Main Document Topics...')
console.print(f'[grey78]5. Generating Document Table of Contents...')
console.print(f'[grey78]6. Generating Document SUMMARY...')
console.print(f'[green1 bold]7. Saving the QnA database into picklefile: {dbfilename}')
SaveFirstData(dbfilename, replies)
console.print('--------------------------------------------------------------------------------------')
console.print(f'✅✅✅COMPLETED ANALYSIS OF DOCUMENT {path} \n \n')
console.print(f'PIPELINE Generated in {maindelta}')
console.print(f'SELECTED MODEL h2o-danube-1.8b-chat.Q5_K_M.gguf  AND  stablelm-zephyr-3b.Q4_K_M.gguf')
console.print('--------------------------------------------------------------------------------------')
finalstring = f'''--------------------------------------------------------------------------------------
PIPELINE Generated in {maindelta}
SELECTED MODEL: h2o-danube-1.8b-chat.Q5_K_M.gguf  AND  stablelm-zephyr-3b.Q4_K_M.gguf
--------------------------------------------------------------------------------------'''
writehistory(logfile,finalstring)
console.print(f'[green1 bold]BYE BYE')

console = Console(width=120)
def print_db(db, title):
  #https://rich.readthedocs.io/en/stable/tables.html
  from rich.table import Table
  table = Table(title=title, show_lines=True)
  table.add_column("ID", justify="left", style="red", width=5)
  table.add_column("Question", justify="left", style="yellow", width=45)
  table.add_column("Answer", justify="left", style="cyan", width=70)
  #table.add_column("Doc_id", justify="center", style="cyan")
  #table.add_column("Box Office", justify="right", style="green")
  i=0
  for hit in db:
    table.add_row(str(i),hit['question'], hit['answer'])
    i += 1
  console.print(table)

print_db(replies, 'DOCUMENT EXPLORATION')
