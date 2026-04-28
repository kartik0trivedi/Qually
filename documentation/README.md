# Qually Documentation

This directory contains documentation for writing an ArXiv article and user manual for Qually.

## Files

- **`arxiv_article_guide.md`**: Comprehensive guide on how to structure and write the ArXiv article
- **`qually_arxiv_template.tex`**: LaTeX template for the main ArXiv article (technical/scientific paper)
- **`qually_user_manual.tex`**: LaTeX template for the standalone user manual (detailed instructions)

## Document Structure

The documentation is split into two separate documents:

### 1. Main ArXiv Article (`qually_arxiv_template.tex`)
- Scientific/technical paper describing Qually
- System architecture and design
- Features and capabilities
- Use cases and examples
- Technical details
- References the user manual (separate document)

### 2. User Manual (`qually_user_manual.tex`)
- Comprehensive step-by-step instructions
- Installation and setup
- Detailed usage guide
- Troubleshooting
- Best practices
- Standalone document with table of contents

## Template Status

Both LaTeX templates have been fixed to compile without errors:

✅ **Fixed Issues:**
- Figure reference commented out (add your architecture diagram when ready)
- Float specifier changed from `[h]` to `[ht]` for better placement
- All 6 providers added to the table (OpenAI, Anthropic, Google, Mistral, Grok, DeepSeek)
- Bibliography changed to manual format (no external .bib file required)
- User manual section removed from main template and created as separate document

## Next Steps

1. **Add Figures**: Create and include:
   - System architecture diagram (for main article)
   - Screenshots of the GUI interface (for user manual)
   - Example workflow diagrams (for user manual)

2. **Fill Content**: Replace placeholder comments with actual content for each section

3. **Add References**: Populate the bibliography with relevant citations

4. **Compile Both Documents**: Use `pdflatex` or your preferred LaTeX compiler:
   ```bash
   pdflatex qually_arxiv_template.tex
   pdflatex qually_user_manual.tex
   ```

## Version Note

The template aligns with Qually version 2.0.0 (LE2) as documented in the main README. Ensure consistency with the [GitHub repository](https://github.com/kartik0trivedi/Qually) when referencing version numbers.

