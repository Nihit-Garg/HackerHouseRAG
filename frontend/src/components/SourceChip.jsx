import { Quote } from "lucide-react"
import { truncate } from "../utils/format"

export default function SourceChip({ source }) {
  return (
    <span className="source-chip" title={source.snippet}>
      <Quote size={11} strokeWidth={2.2} />
      <span>{truncate(source.snippet, 60)}</span>
    </span>
  )
}
