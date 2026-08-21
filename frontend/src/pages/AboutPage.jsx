import { CheckCircle2, Info, Keyboard, ListChecks, Search, Sparkles } from "lucide-react"
import IconCircle from "../components/IconCircle"

const STEPS = [
  {
    icon: Keyboard,
    title: "1. Ask",
    text: "Type a question or speak it — Lumina transcribes voice input before doing anything else.",
  },
  {
    icon: Search,
    title: "2. Search",
    text: "Lumina searches only its own curated knowledge base, combining keyword and meaning-based search.",
  },
  {
    icon: ListChecks,
    title: "3. Check",
    text: "Every question and every answer passes through safety checks that catch unsafe, off-topic, or unsupported claims.",
  },
  {
    icon: CheckCircle2,
    title: "4. Answer",
    text: "A concise answer is returned, grounded only in the retrieved passages — with sources attached.",
  },
]

export default function AboutPage() {
  return (
    <div>
      <div className="page-header">
        <div className="page-heading">
          <IconCircle icon={Info} />
          <h1>Understanding Lumina AI</h1>
        </div>
      </div>

      <p className="page-subtitle">
        A focused, secure knowledge retrieval system designed for precision and reliability. We
        prioritize verified information over conversational generation.
      </p>

      <div className="card">
        <div className="card-heading">
          <IconCircle icon={Sparkles} />
          <h2>What is Lumina AI?</h2>
        </div>
        <p>
          Lumina is not a general-purpose conversational chatbot. It is a highly specialized Q&amp;A
          tool connected to a curated, offline knowledge base. It does not browse the internet or
          hallucinate facts to keep a conversation going.
        </p>
        <p>
          Its sole purpose is to retrieve accurate, contextually relevant information from its
          verified dataset and present it clearly. If the answer isn't in the dataset, Lumina won't
          guess.
        </p>
      </div>

      <div className="card">
        <div className="card-heading">
          <IconCircle icon={ListChecks} />
          <h2>How It Works</h2>
        </div>
        <div className="steps-grid">
          {STEPS.map(({ icon: Icon, title, text }) => (
            <div className="step" key={title}>
              <div className="step-icon">
                <Icon size={19} strokeWidth={2.2} />
              </div>
              <div>
                <p className="step-title">{title}</p>
                <p className="step-text">{text}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
