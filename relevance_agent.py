# https://habr.com/ru/articles/864646/

import sys
import re

sys.path.append('../../')
from llm_secrets import *
import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import TavilySearchResults
from langchain.agents import load_tools


class RelevanceAgent:
    def __init__(self, model_name: str, system_prompt: str):
        print('LLM:', model_name)
        self.system_prompt = system_prompt

        # Tavily для поиска в интернете
        self.search_tool = TavilySearchResults(max_results=10,
                                  include_answer=True,
                                  include_raw_content=False, tavily_api_key=TAVILY_API_KEY)
        # Human tool для взаимодействия с пользователем
        # self.human_tool = load_tools(["human"])[0]
        # self.tools = [self.search_tool, self.human_tool]

        self.tools = [self.search_tool]

        # Настраиваем модель и биндим к ней инструменты
        self.llm = ChatOpenAI(model=model_name, max_tokens=500, api_key=OPENAI_TOKEN, base_url=OPENAI_PROXY_URL, timeout=30).bind_tools(self.tools)

        # встроенная в langgraph нода вызова инструментов
        self.tool_node = ToolNode(self.tools)

        workflow = StateGraph(MessagesState)

        # задаём ноды
        workflow.add_node("gather_data_node", self.gather_data)
        workflow.add_node("tools", self.tool_node)

        # задаём переходы между нодами
        # входная нода - gather_data_node
        workflow.set_entry_point("gather_data_node")
        # после gather_data_node вызываем should_continue,
        # чтобы определить что делать дальше
        workflow.add_conditional_edges("gather_data_node",
                                       self.should_continue,
                                       ["tools", END])
        # после вызова инструментов всегда возвращаемся к ноде LLM,
        # чтобы отдать ей результат вызова инструментов
        workflow.add_edge("tools", "gather_data_node")

        self.graph = workflow.compile()


    # функция, которая определяет нужно ли вызывать инструменты
    # или результат уже получен
    def should_continue(self, state: MessagesState):
        print('SHOULD_CONTINUE')
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END


    # функция для ноды взаимодейтсвия с LLM
    def gather_data(self, state: MessagesState):
        print('GATHER_DATA_START')
        messages = state["messages"]

        messages.append(SystemMessage(content=self.system_prompt))

        response = self.llm.invoke(messages)
        print('GATHER_DATA_SUCCESS')

        # информация для отладки
        print(json.dumps(response.tool_calls, indent=2, ensure_ascii=False))
        print(json.dumps(response.content, indent=2, ensure_ascii=False))

        return {"messages": [response]}



    def run_agent(self, prompt: str):
        input_messages = [HumanMessage(prompt)]
        output = self.graph.invoke({"messages": input_messages})
        return output

    def parse_output(self, output, enable_llm_comments=True) -> list:
        answer = output['messages'][-1].content

        if enable_llm_comments:
            # Достаём comment (всё между "comment": " и следующей кавычкой ")
            comment_match = re.search(r'"comment":\s*"([^"]+)"', answer)
            comment = comment_match.group(1) if comment_match else None

            # Достаём score (число после "score": )
            score_match = re.search(r'"score":\s*(\d+)', answer)
            score = int(score_match.group(1)) if score_match else None
        else:
            score_match = re.search(r'(\d+)', answer)
            score = int(score_match.group(1)) if score_match else None
            comment = '-'


        search_queries = []
        for message in output['messages']:
            if 'tool_calls' in message.additional_kwargs:  # type(message) == langchain_core.messages.ai.AIMessage:
                search_queries.extend(
                    [x['function']['arguments'][10:-2] for x in message.additional_kwargs['tool_calls']])
        return score, comment, search_queries


if __name__=='__main__':
    system_prompt = '''
            Ты – специалист по поиску организаций в интернете по широким запросам пользователей. Пользователь отправляет широкий запрос (например, "ресторан с верандой" или "романтичный джаз-бар"). Такие запросы называются "рубричными": пользователь здесь ищет не конкретную организацию, а идёт в Яндекс.Карты для поиска и выбора мест.

            Твоя задача -  оценить релевантность запросу конкретной организации, информация о которой приведена в сообщении пользователя. Наличие дополнительных услуг или товаров не должно снижать оценку, главное - чтобы в организации было то, что нужно пользователю. Не важно, получена информация о том, что нужно пользователю, из исходного описания или из результатов поиска.

            Для поиска данных о компании используй инструмент TavilySearchResults. При использовании поиска ищи каждый параметр отдельным поисковым запросом.
            Продолжай пользоваться поиском до тех пор, пока не соберешь все данные.

            Ответь JSON в формате {{"comment": комментарий, почему организация является релевантной или не релевантной, "score": оценка релевантности организации запросу пользователя от 0 до 100 (100 - точно релевантна, 0 - точно не релевантна, 67 - релевантна на 67%)}}.
            Никакий других данных в ответе быть не должно, только JSON, который можно распарсить
            '''

    MODEL_NAME = 'deepseek/deepseek-chat'
    # relevance_agent = RelevanceAgent(model_name='deepseek/deepseek-chat', system_prompt=system_prompt)
    relevance_agent = RelevanceAgent(model_name=MODEL_NAME, system_prompt=system_prompt)
    # изначальное описание ситуации от клиента
    prompt = """
        Вот описание ситуации от клиента:
        Я купил абонемент в спортзале Wordclass на ленинском 109 на год.
        Через месяц я захотел его вернуть и на ресепшене мне отказали.
        """

    output = relevance_agent.run_agent(prompt)
    print(output)