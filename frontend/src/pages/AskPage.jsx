import { Fragment, useEffect, useRef, useState } from "react"
import { Loader2, Mic, Paperclip, Send, Square, Image as ImageIcon } from "lucide-react"
import { askAudio, askText } from "../api"
import { useHistory } from "../hooks/useHistory"
import { buildEntry } from "../utils/entries"
import { formatDuration } from "../utils/format"
import MessageBubble from "../components/MessageBubble"
import AnswerCard from "../components/AnswerCard"

const EXAMPLE_QUESTIONS = [
  "What is photosynthesis?",
  "How does FAISS similarity search work?",
  "What causes earthquakes?",
  "How does a vaccine work?",
]

const AUDIO_ACCEPT = ".wav,.mp3,.ogg,.flac,.m4a,.webm"

export default function AskPage() {
  const { entries, addEntry } = useHistory()
  const [inputValue, setInputValue] = useState("")
  const [pending, setPending] = useState(null)
  const [error, setError] = useState(null)
  const [isRecording, setIsRecording] = useState(false)
  const [recordSeconds, setRecordSeconds] = useState(0)

  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])
  const fileInputRef = useRef(null)
  const conversationEndRef = useRef(null)

  const chatEntries = [...entries].reverse()

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatEntries.length, pending])

  useEffect(() => {
    if (!isRecording) return
    const interval = setInterval(() => setRecordSeconds((seconds) => seconds + 1), 1000)
    return () => clearInterval(interval)
  }, [isRecording])

  async function submitText(rawText) {
    const text = rawText.trim()
    if (!text || pending) return

    setInputValue("")
    setError(null)
    setPending({ question: text, isAudio: false })

    try {
      const result = await askText(text)
      addEntry(buildEntry(result))
    } catch (err) {
      setError(err.message)
    } finally {
      setPending(null)
    }
  }

  async function submitAudio(file) {
    setError(null)
    setPending({ question: null, isAudio: true })

    try {
      const result = await askAudio(file)
      addEntry(buildEntry(result))
    } catch (err) {
      setError(err.message)
    } finally {
      setPending(null)
    }
  }

  async function startRecording() {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []

      const recorder = new MediaRecorder(stream)
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }
      recorder.onstop = () => {
        streamRef.current?.getTracks().forEach((track) => track.stop())
        const blob = new Blob(chunksRef.current, { type: "audio/webm" })
        submitAudio(new File([blob], "voice-message.webm", { type: "audio/webm" }))
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setRecordSeconds(0)
      setIsRecording(true)
    } catch {
      setError("Microphone access was denied.")
    }
  }

  function stopRecording() {
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
  }

  function handleFileChange(event) {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (file) submitAudio(file)
  }

  function handleKeyDown(event) {
    if (event.key === "Enter") submitText(inputValue)
  }

  function handleTrailingClick() {
    if (inputValue.trim()) {
      submitText(inputValue)
    } else if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  const isBusy = Boolean(pending)
  const TrailingIcon = isRecording ? Square : inputValue.trim() ? Send : Mic

  return (
    <div className="ask-page">
      <div className="ask-header">
        <h1>Ask Lumina</h1>
        <p>How can I help you explore your knowledge today?</p>
        {chatEntries.length === 0 && !pending && (
          <div className="ask-examples">
            {EXAMPLE_QUESTIONS.map((question) => (
              <button key={question} className="example-chip" onClick={() => submitText(question)}>
                {question}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="conversation">
        {chatEntries.map((entry) => (
          <Fragment key={entry.id}>
            <MessageBubble text={entry.question} />
            <AnswerCard entry={entry} />
          </Fragment>
        ))}

        {pending && (
          <Fragment>
            <MessageBubble text={pending.question || "Voice message"} />
            <div className="loading-row">
              <Loader2 size={16} />
              {pending.isAudio ? "Transcribing your voice…" : "Searching knowledge base…"}
            </div>
          </Fragment>
        )}

        <div ref={conversationEndRef} />
      </div>

      <div className="input-bar">
        <button
          className="icon-button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isBusy || isRecording}
          title="Upload an audio file"
        >
          <Paperclip size={17} strokeWidth={2.2} />
        </button>
        <button className="icon-button" disabled title="Image attachments aren't supported yet">
          <ImageIcon size={17} strokeWidth={2.2} />
        </button>

        {isRecording ? (
          <div className="record-indicator">
            <span className="record-dot" />
            Recording {formatDuration(recordSeconds)}
          </div>
        ) : (
          <input
            type="text"
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message or ask by voice…"
            disabled={isBusy}
          />
        )}

        <button
          className={`trailing-button${isRecording ? " recording" : ""}`}
          onClick={handleTrailingClick}
          disabled={isBusy}
        >
          <TrailingIcon size={18} strokeWidth={2.2} />
        </button>

        <input
          ref={fileInputRef}
          type="file"
          accept={AUDIO_ACCEPT}
          hidden
          onChange={handleFileChange}
        />
      </div>

      {error && <p className="ask-error">{error}</p>}
      <p className="ask-caption">Answers are generated only from verified sources — nothing is invented.</p>
    </div>
  )
}
