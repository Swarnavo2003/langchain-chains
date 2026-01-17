from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1 = ChatOpenAI(model="gpt-4o-mini")
model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

parser = StrOutputParser()

class FeedBack(BaseModel):
  sentiment: Literal['positive', 'negative'] = Field(description='The sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
  template="Classify the sentiment of the following deefback text into positive or negative \n {feedback}\n {format_instructions}",
  input_variables=['feedback'],
  partial_variables={'format_instructions': parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model1 | parser2

prompt2 = PromptTemplate(
  template="Write an appropriate response to this positive feedbacks \n {feedback}",
  input_variables=['feedback'],
)

prompt3 = PromptTemplate(
  template="Write an appropriate response to this negative feedbacks \n {feedback}",
  input_variables=['feedback'],
)

branch_chain = RunnableBranch(
  (lambda x: x.sentiment == 'positive', prompt2 | model1 | parser),
  (lambda x: x.sentiment == 'negative', prompt3 | model1 | parser),
  RunnableLambda(lambda x: "could not find sentiment"),
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "I like this product"})

print(result)

chain.get_graph().print_ascii()