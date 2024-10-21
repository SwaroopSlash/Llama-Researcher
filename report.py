from datetime import datetime, timezone

from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.base import PromptTemplate

from llama_index.core.prompts.base import PromptTemplate, PromptType 


async def generate_report_from_context(query: str, context: str, llm: LLM) -> str:
    prompt = PromptTemplate(
        """Information:
--------------------------------
{context}
--------------------------------
Using the above information, answer the following query or task: "{question}" in a detailed report --
The report should focus on the answer to the query, should be well structured, informative, 
in-depth, and comprehensive, with facts and numbers if available and at least {total_words} words.
You should strive to write the report as long as you can using all relevant and necessary information provided.

**Report Title and Introduction:**

   - Begin the report with a title aligned at the center of the document. The title should be formatted in bold and use a larger font size.
   - Use a proper Title for the report.

**Introduction related to the section Summary:**
- The report should start with a concise summary section.
- Label the summary section as "Summary" (in bold).
- This section should provide a brief overview of the report, highlighting key points but avoiding detailed content.
- Place this summary before the numbered sections of the report.
- Do not include complex formatting or subsections in the summary.

Please follow all of the following guidelines in your report:

- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST write the report with markdown syntax and {report_format} format.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of the paragraph that references them like this: ([in-text citation](Website Name)).


Please follow the instructions below to ensure correct numbering, indentation, and visual clarity for sections, subsections, and further content:

1. **Main Sections (Standard Roman Numerals):**
   - Use Roman numerals (I, II, III, …) for the main sections.
   - The numbering should be in a sequential manner.
   - Titles should be aligned with the left border, no indentation. Font size should be 14.
   - Leave one blank line before and after each section.

   Example:
   I. Main Section Title

   Content for the main section goes here...

2. **Further Subsections:**
   - For additional subsections or details, use bullet  points (e.g., `-` or `*`) .
   - Ensure that the bullet points are properly nested under the subsection titles.
   - Indent the bullet points by an additional `Tab` (approx. 4 spaces) from the subsection title.


3. **General Content Guidelines:**
   - Ensure all content is properly indented according to its section or subsection.
   - Maintain consistent spacing between paragraphs, bullet points, and lists.
   - Lists/steps within subsections should follow the appropriate hierarchy and indentation.

Please follow the instructions below for the "References" section:

You MUST include markdown hyperlinks to the relevant URLs wherever they are referenced in the report in a separate section "References" at the end of the report:

eg:
    References:
    [1] Author, A. A. (Year of the report, Month Date). Title of the web page. Website Name. (URL to Source) \n
    [2] Author, C. B. (Year of the report, Month Date). Title of the web page. Website Name. (URL to Source) \n
    and so on....

- Make sure to not add duplicated sources, but only one reference for each. Also, every URL should be markdown hyperlinked: [url website](url).
- You should only include URLs whose maximum data is used. Keep the number of Reference URLs concise.
- URLs with markdown hyperlinks is an important aspect of the report; strictly follow {report_format} format in the section. Be very mindful while drafting this section.

Please do your best; this report is going to be used in actual work, so follow all standard practices.
Assume that the current date is {date_today} and mention the date below the heading in the report with Bold Font. The date is an important aspect of the report.
"""

    )

    prompt_template = PromptTemplate(
        template=prompt.template,
        prompt_type=PromptType.QUESTION_ANSWER,
    )

    # Pass the PromptTemplate object to apredict
    response = await llm.apredict(
        prompt_template,
        context = context,
        question = query,        
        total_words = 2000,
        report_format = "APA",
        date_today = datetime.now(timezone.utc).strftime("%B %d, %Y"),

    )


    print("\n> Done generating report\n")

    return response
