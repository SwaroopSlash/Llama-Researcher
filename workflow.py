from typing import List, Any
import subprocess

from llama_index.core.schema import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from markdown_pdf import MarkdownPdf, Section

from subquery import get_sub_queries
from tavily import get_urls_from_tavily_search
from scraper import get_scraped_docs_from_urls
from compress import get_compressed_context
from report import generate_report_from_context


class SubQueriesCreatedEvent(Event):
    sub_queries: List[str]


class ToProcessSubQueryEvent(Event):
    sub_query: str


class ToScrapeWebContentsEvent(Event):
    sub_query: str
    urls: List[str]


class DocsScrapedEvent(Event):
    sub_query: str
    docs: List[Document]


class ToCombineContextEvent(Event):
    sub_query: str
    context: str


class ReportPromptCreatedEvent(Event):
    context: str


class LLMResponseEvent(Event):
    response: str


class ResearchAssistantWorkflow(Workflow):
    def __init__(
        self,
        *args: Any,
        llm: LLM,
        embed_model: BaseEmbedding,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.embed_model = embed_model
        self.visited_urls: set[str] = set()

    @step
    async def create_sub_queries(
        self, ctx: Context, ev: StartEvent
    ) -> SubQueriesCreatedEvent:
        query = ev.query
        await ctx.set("query", query)
        sub_queries = await get_sub_queries(query, self.llm)
        await ctx.set("num_sub_queries", len(sub_queries))
        return SubQueriesCreatedEvent(sub_queries=sub_queries)

    @step
    async def deligate_sub_queries(
        self, ev: SubQueriesCreatedEvent
    ) -> ToProcessSubQueryEvent:
        for sub_query in ev.sub_queries:
            self.send_event(ToProcessSubQueryEvent(sub_query=sub_query))
        return None

    @step(num_workers=5)
    async def get_urls_for_subquery(
        self, ev: ToProcessSubQueryEvent
    ) -> ToScrapeWebContentsEvent:
        sub_query = ev.sub_query
        print(f"\n> Getting urls for sub query: {sub_query}\n")
        urls = await get_urls_from_tavily_search(sub_query)
        new_urls = []
        for url in urls:
            if url not in self.visited_urls:
                self.visited_urls.add(url)
                new_urls.append(url)
        return ToScrapeWebContentsEvent(sub_query=sub_query, urls=new_urls)

    @step(num_workers=5)
    async def scrape_web_contents(
        self, ev: ToScrapeWebContentsEvent
    ) -> DocsScrapedEvent:
        sub_query = ev.sub_query
        urls = ev.urls
        print(f"\n> Scraping web contents for sub query: {sub_query}\n")
        docs = get_scraped_docs_from_urls(urls)
        return DocsScrapedEvent(sub_query=sub_query, docs=docs)

    @step(num_workers=5)
    async def compress_docs(self, ev: DocsScrapedEvent) -> ToCombineContextEvent:
        sub_query = ev.sub_query
        docs = ev.docs
        print(f"\n> Compressing docs for sub query: {sub_query}\n")
        compressed_context = await get_compressed_context(
            sub_query, docs, self.embed_model
        )
        return ToCombineContextEvent(sub_query=sub_query, context=compressed_context)

    @step
    async def combine_contexts(
        self, ctx: Context, ev: ToCombineContextEvent
    ) -> ReportPromptCreatedEvent:
        events = ctx.collect_events(
            ev, [ToCombineContextEvent] * await ctx.get("num_sub_queries")
        )
        if events is None:
            return None

        context = ""

        for event in events:
            context += (
                f'Research findings for topic "{event.sub_query}":\n{event.context}\n\n'
            )

        return ReportPromptCreatedEvent(context=context)

    @step
    async def write_report(self, ev: ReportPromptCreatedEvent) -> StopEvent:
        context = ev.context
        query = await self.context.get("query")
        print(f"\n> Writing report. This will take a few minutes...\n")
        report = await generate_report_from_context(query, context, self.llm)
        pdf = MarkdownPdf()
        pdf.add_section(Section(report, toc=False))
        pdf.save("report.pdf")
        print("\n> Done writing report to report.pdf! Trying to open the file...\n")
        subprocess.run(["open", "report.pdf"])
        return StopEvent(result="Done")
