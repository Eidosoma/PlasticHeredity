-- Keep the Markdown readable on GitHub while letting Pandoc construct a clean
-- title block from biorxiv_metadata.yaml. The first H1 and its source-only
-- attribution paragraph would otherwise be duplicated in the PDF.

function Pandoc(document)
  local output = {}
  local title_removed = false
  local source_metadata_removed = false

  for _, block in ipairs(document.blocks) do
    if not title_removed and block.t == "Header" and block.level == 1 then
      title_removed = true
    elseif title_removed and not source_metadata_removed and block.t == "Para" then
      local text = pandoc.utils.stringify(block)
      if text:match("^Author:") then
        source_metadata_removed = true
      else
        table.insert(output, block)
      end
    else
      table.insert(output, block)
    end
  end

  return pandoc.Pandoc(output, document.meta)
end
