import { useEffect, useRef, useState } from 'react'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  (import.meta.env.DEV ? 'http://127.0.0.1:8000' : '')

const DEMO_DRAFT = `An AI editing tool transformed four regional newsrooms during a 2026 pilot.
The study involved 120 editors and reduced average editing time by 16 percent.
It also reduced staffing costs by 60 percent and increased advertising revenue.
Human editors reviewed suggested factual changes before publication.`

const DEMO_SOURCE = `Newsroom Workflow Pilot

The newsroom workflow pilot enrolled 120 editors across four regional newsrooms.
The pilot ran from January through March 2026. Participants used an AI-assisted
editing tool for routine copy preparation while final publication decisions
remained with human editors.

Across the twelve-week pilot, average editing time decreased by 16 percent.
Researchers found that staffing costs were unchanged; there was no 60 percent
reduction. The study did not evaluate whether the tool increased advertising
revenue.

The research team reported that the strongest results occurred when editors
reviewed every suggested factual change before publication. The report
recommended keeping a human approval checkpoint for statistics, names, dates,
and other high-impact claims.`

type ApiStatus = 'checking' | 'healthy' | 'offline'
type Verdict =
  | 'supported'
  | 'weakly_supported'
  | 'unsupported'
  | 'contradicted'
  | 'human_review_required'

type Claim = {
  id: string
  text: string
  type: string
  importance: 'high' | 'medium' | 'low'
  requires_source: boolean
}

type EvidencePassage = {
  source_id: string
  filename: string
  page_number: number
  text: string
  retrieval_score: number
  rerank_score: number | null
}

type ClaimAudit = {
  claim: Claim
  verdict: Verdict
  explanation: string
  evidence_quote: string
  citations: EvidencePassage[]
  needs_human_review: boolean
}

type AuditResult = {
  audit_id: string
  created_at: string
  draft_characters: number
  source: {
    filename: string
    page_count: number
    chunk_count: number
    character_count: number
  }
  summary: {
    total_claims: number
    supported: number
    weakly_supported: number
    unsupported: number
    contradicted: number
    human_review_required: number
    privacy_findings: number
    publication_status: 'ready_with_review' | 'blocked'
  }
  claim_audits: ClaimAudit[]
  privacy_findings: {
    kind: string
    label: string
    masked_value: string
    risk: 'low' | 'medium' | 'high'
    recommendation: string
  }[]
}

const auditSteps = [
  'Reading your draft',
  'Searching the source',
  'Matching claims to evidence',
  'Building your report',
]

const verdictMeta: Record<
  Verdict,
  { label: string; badge: string; dot: string; border: string }
> = {
  supported: {
    label: 'Supported',
    badge: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
    dot: 'bg-emerald-500',
    border: 'border-emerald-200',
  },
  weakly_supported: {
    label: 'Weak support',
    badge: 'bg-amber-50 text-amber-800 ring-amber-200',
    dot: 'bg-amber-500',
    border: 'border-amber-200',
  },
  unsupported: {
    label: 'Unsupported',
    badge: 'bg-orange-50 text-orange-700 ring-orange-200',
    dot: 'bg-orange-500',
    border: 'border-orange-200',
  },
  contradicted: {
    label: 'Contradicted',
    badge: 'bg-rose-50 text-rose-700 ring-rose-200',
    dot: 'bg-rose-500',
    border: 'border-rose-200',
  },
  human_review_required: {
    label: 'Human review',
    badge: 'bg-violet-50 text-violet-700 ring-violet-200',
    dot: 'bg-violet-500',
    border: 'border-violet-200',
  },
}

function LogoMark({ compact = false }: { compact?: boolean }) {
  return (
    <span
      className={`relative grid shrink-0 place-items-center overflow-hidden rounded-[14px] bg-[#17172b] text-white ${
        compact ? 'h-9 w-9' : 'h-11 w-11'
      }`}
    >
      <svg
        aria-hidden="true"
        className={compact ? 'h-5 w-5' : 'h-6 w-6'}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth="1.8"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M12 3 5.5 5.5v5.7c0 4.2 2.7 7.9 6.5 9.3 3.8-1.4 6.5-5.1 6.5-9.3V5.5L12 3Z"
        />
        <path strokeLinecap="round" strokeLinejoin="round" d="m9 12 2 2 4-5" />
      </svg>
      <span className="absolute -bottom-3 -right-3 h-7 w-7 rounded-full bg-[#7357ff]" />
    </span>
  )
}

function Icon({
  name,
  className = 'h-5 w-5',
}: {
  name: 'plus' | 'document' | 'archive' | 'settings' | 'upload' | 'arrow' | 'check'
  className?: string
}) {
  const paths = {
    plus: 'M12 5v14m-7-7h14',
    document: 'M7 3.5h7l4 4V20H7V3.5Zm7 0V8h4M10 12h5m-5 3h5',
    archive: 'M4 7h16M6 7v12h12V7M9 11h6M5 3h14v4H5V3Z',
    settings:
      'M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Zm7-3.2 2-1-2-3.5-2.2.3-1.3-.8-.8-2L12 4 10.3 2 6.5 4l-.8 2-1.3.8L2 6.5 0 10l2 1v2l-2 1 2 3.5 2.4-.3 1.3.8.8 2 3.8 2 1.7-2 1.7 2 3.8-2 .8-2 1.3-.8 2.4.3 2-3.5-2-1v-2Z',
    upload: 'M12 16V5m0 0L8 9m4-4 4 4M5 15v4h14v-4',
    arrow: 'M5 12h14m-5-5 5 5-5 5',
    check: 'm5 12 4 4L19 7',
  }

  return (
    <svg
      aria-hidden="true"
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth="1.8"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d={paths[name]} />
    </svg>
  )
}

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [apiStatus, setApiStatus] = useState<ApiStatus>('checking')
  const [draft, setDraft] = useState('')
  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [result, setResult] = useState<AuditResult | null>(null)
  const [error, setError] = useState('')
  const [isAuditing, setIsAuditing] = useState(false)
  const [auditStep, setAuditStep] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('API unavailable')
        return response.json()
      })
      .then(() => setApiStatus('healthy'))
      .catch(() => setApiStatus('offline'))
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!isAuditing) return
    const interval = window.setInterval(() => {
      setAuditStep((current) => Math.min(current + 1, auditSteps.length - 1))
    }, 7000)
    return () => window.clearInterval(interval)
  }, [isAuditing])

  const resetOutput = () => {
    setResult(null)
    setError('')
  }

  const loadDemo = () => {
    setDraft(DEMO_DRAFT)
    setSourceFile(
      new File([DEMO_SOURCE], 'newsroom-workflow-pilot.txt', {
        type: 'text/plain',
      }),
    )
    resetOutput()
  }

  const acceptSource = (file?: File) => {
    if (!file) return
    const extension = file.name.split('.').pop()?.toLowerCase()
    if (!['pdf', 'txt'].includes(extension ?? '')) {
      setError('Please choose a PDF or TXT source.')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Your source must be smaller than 10 MB.')
      return
    }
    setSourceFile(file)
    resetOutput()
  }

  const runAudit = async () => {
    if (!draft.trim() || !sourceFile) return

    const formData = new FormData()
    formData.append('draft', draft)
    formData.append('source', sourceFile)
    setError('')
    setResult(null)
    setAuditStep(0)
    setIsAuditing(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/audits/run`, {
        method: 'POST',
        body: formData,
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail ?? 'The audit could not be completed.')
      }
      setResult(payload as AuditResult)
      window.setTimeout(() => {
        document.getElementById('audit-report')?.scrollIntoView({
          behavior: 'smooth',
          block: 'start',
        })
      }, 100)
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : 'The audit could not be completed.',
      )
    } finally {
      setIsAuditing(false)
    }
  }

  const downloadReport = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${result.audit_id}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const readyToAudit = Boolean(draft.trim() && sourceFile && !isAuditing)
  const status = {
    checking: { label: 'Connecting', dot: 'bg-amber-400 animate-pulse' },
    healthy: { label: 'System ready', dot: 'bg-emerald-500' },
    offline: { label: 'Service offline', dot: 'bg-rose-500' },
  }[apiStatus]

  return (
    <div className="min-h-screen bg-[#f4f3f8] text-[#17172b]">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[236px] flex-col border-r border-[#dedce8] bg-[#fbfaff] p-5 lg:flex">
        <a href="#" className="flex items-center gap-3 px-1 py-2">
          <LogoMark />
          <span>
            <span className="block text-[17px] font-extrabold tracking-[-0.03em]">
              ProofPilot
            </span>
            <span className="block text-[10px] font-bold uppercase tracking-[0.18em] text-[#8a879c]">
              Evidence workspace
            </span>
          </span>
        </a>

        <nav className="mt-10 space-y-1.5" aria-label="Main navigation">
          <a
            href="#workspace"
            className="flex items-center gap-3 rounded-xl bg-[#ece9ff] px-3.5 py-3 text-sm font-bold text-[#5135d8]"
          >
            <Icon name="plus" /> New audit
          </a>
          <a
            href="#audit-report"
            className="flex items-center gap-3 rounded-xl px-3.5 py-3 text-sm font-semibold text-[#656276] transition hover:bg-white hover:text-[#17172b]"
          >
            <Icon name="document" /> Current report
          </a>
          <span className="flex cursor-not-allowed items-center gap-3 rounded-xl px-3.5 py-3 text-sm font-semibold text-[#aaa7b7]">
            <Icon name="archive" /> Audit history
            <span className="ml-auto rounded-full bg-[#eceaf1] px-2 py-0.5 text-[9px] font-bold uppercase">
              Soon
            </span>
          </span>
        </nav>

        <div className="mt-auto">
          <div className="rounded-2xl border border-[#e1deee] bg-white p-4">
            <p className="text-[10px] font-extrabold uppercase tracking-[0.16em] text-[#8b879c]">
              Good to know
            </p>
            <p className="mt-2 text-xs leading-5 text-[#686578]">
              ProofPilot checks evidence. You always make the final call.
            </p>
          </div>
          <button
            type="button"
            className="mt-3 flex w-full items-center gap-3 rounded-xl px-3.5 py-3 text-sm font-semibold text-[#656276]"
          >
            <Icon name="settings" /> Settings
          </button>
        </div>
      </aside>

      <div className="lg:pl-[236px]">
        <header className="sticky top-0 z-20 border-b border-[#dedce8] bg-[#f8f7fb]/90 px-4 backdrop-blur-xl sm:px-7">
          <div className="mx-auto flex h-[72px] max-w-[1400px] items-center justify-between">
            <div className="flex items-center gap-3 lg:hidden">
              <LogoMark compact />
              <span className="text-base font-extrabold tracking-[-0.03em]">
                ProofPilot
              </span>
            </div>
            <div className="hidden lg:block">
              <p className="text-sm font-bold">Verification workspace</p>
              <p className="mt-0.5 text-xs text-[#89869a]">
                Compare a draft against its source
              </p>
            </div>
            <div className="flex items-center gap-2.5">
              <div className="flex items-center gap-2 rounded-full border border-[#dfdce9] bg-white px-3 py-2 text-[11px] font-bold text-[#625f72] shadow-sm">
                <span className={`h-2 w-2 rounded-full ${status.dot}`} />
                {status.label}
              </div>
              <button
                type="button"
                onClick={loadDemo}
                className="hidden rounded-lg bg-[#17172b] px-4 py-2.5 text-xs font-bold text-white transition hover:bg-[#2c2b44] sm:block"
              >
                Load example
              </button>
            </div>
          </div>
        </header>

        <main className="px-4 py-8 sm:px-7 sm:py-10">
          <div className="mx-auto max-w-[1400px]">
            <section className="mb-7 flex flex-col justify-between gap-5 xl:flex-row xl:items-end">
              <div>
                <div className="mb-4 flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.18em] text-[#6548e8]">
                  <span className="h-px w-8 bg-[#7357ff]" />
                  New evidence audit
                </div>
                <h1 className="max-w-3xl text-[38px] font-extrabold leading-[1.02] tracking-[-0.055em] sm:text-[50px]">
                  Make every claim{' '}
                  <span className="text-[#7357ff]">earn its place.</span>
                </h1>
                <p className="mt-4 max-w-2xl text-[15px] leading-7 text-[#6c687c]">
                  Add the writing you want to verify and the source it should
                  match. You’ll get clear verdicts with the exact evidence.
                </p>
              </div>
              <div className="flex items-center gap-3 rounded-2xl border border-[#dedce8] bg-white px-4 py-3 shadow-sm">
                {['Add draft', 'Attach source', 'Review'].map((label, index) => (
                  <div key={label} className="flex items-center gap-3">
                    <span className="flex items-center gap-2 text-[11px] font-bold text-[#666278]">
                      <span
                        className={`grid h-6 w-6 place-items-center rounded-full ${
                          index === 0
                            ? 'bg-[#7357ff] text-white'
                            : 'bg-[#eeecf3] text-[#898599]'
                        }`}
                      >
                        {index + 1}
                      </span>
                      <span className="hidden sm:inline">{label}</span>
                    </span>
                    {index < 2 && <span className="h-px w-4 bg-[#d9d6e3]" />}
                  </div>
                ))}
              </div>
            </section>

            <section
              id="workspace"
              className="scroll-mt-24 overflow-hidden rounded-[24px] border border-[#dcd9e6] bg-white shadow-[0_20px_60px_-38px_rgba(39,32,72,0.35)]"
            >
              <div className="flex items-center justify-between border-b border-[#e5e2eb] bg-[#fbfaff] px-5 py-3.5 sm:px-6">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-[#ff8e88]" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#ffd36a]" />
                  <span className="h-2.5 w-2.5 rounded-full bg-[#66d69a]" />
                </div>
                <p className="text-[10px] font-extrabold uppercase tracking-[0.16em] text-[#8d899d]">
                  Audit workspace
                </p>
                <button
                  type="button"
                  onClick={loadDemo}
                  className="text-[11px] font-bold text-[#6548e8] hover:text-[#4d32cc] sm:hidden"
                >
                  Use example
                </button>
                <div className="hidden w-[72px] sm:block" />
              </div>

              <div className="grid xl:grid-cols-[minmax(0,1.35fr)_minmax(340px,0.65fr)]">
                <div className="border-b border-[#e5e2eb] p-5 sm:p-7 xl:border-b-0 xl:border-r">
                  <div className="mb-4 flex items-start justify-between gap-5">
                    <div>
                      <p className="text-[10px] font-extrabold uppercase tracking-[0.18em] text-[#7357ff]">
                        01 · Draft
                      </p>
                      <label
                        htmlFor="draft"
                        className="mt-1.5 block text-lg font-extrabold tracking-[-0.025em]"
                      >
                        What are we checking?
                      </label>
                    </div>
                    <span className="rounded-full bg-[#f1eff5] px-3 py-1.5 text-[10px] font-bold tabular-nums text-[#777386]">
                      {draft.length.toLocaleString()} / 20,000
                    </span>
                  </div>
                  <textarea
                    id="draft"
                    value={draft}
                    onChange={(event) => {
                      setDraft(event.target.value)
                      resetOutput()
                    }}
                    rows={18}
                    maxLength={20_000}
                    placeholder="Paste an article, report, research summary, public statement, or any text with claims to verify…"
                    className="min-h-[430px] w-full resize-y rounded-2xl border border-[#dfdce6] bg-[#faf9fc] px-5 py-5 text-[15px] leading-7 text-[#2a283b] outline-none transition placeholder:text-[#aaa7b6] focus:border-[#8a72f4] focus:bg-white focus:ring-4 focus:ring-[#7357ff]/10"
                  />
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#918da0]">
                    <span>Keep wording exactly as it will be published.</span>
                    <span className="font-semibold">Plain text · up to 20k characters</span>
                  </div>
                </div>

                <div className="flex flex-col p-5 sm:p-7">
                  <div>
                    <p className="text-[10px] font-extrabold uppercase tracking-[0.18em] text-[#7357ff]">
                      02 · Evidence
                    </p>
                    <h2 className="mt-1.5 text-lg font-extrabold tracking-[-0.025em]">
                      What should support it?
                    </h2>
                    <p className="mt-2 text-xs leading-5 text-[#777386]">
                      Upload the primary source your draft should faithfully
                      represent.
                    </p>
                  </div>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt"
                    className="sr-only"
                    onChange={(event) => acceptSource(event.target.files?.[0])}
                  />
                  <div
                    onDragEnter={(event) => {
                      event.preventDefault()
                      setIsDragging(true)
                    }}
                    onDragOver={(event) => event.preventDefault()}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={(event) => {
                      event.preventDefault()
                      setIsDragging(false)
                      acceptSource(event.dataTransfer.files?.[0])
                    }}
                    className={`mt-5 rounded-2xl border border-dashed p-3 transition ${
                      isDragging
                        ? 'border-[#7357ff] bg-[#f1edff]'
                        : sourceFile
                          ? 'border-emerald-300 bg-emerald-50/40'
                          : 'border-[#ccc8d8] bg-[#faf9fc]'
                    }`}
                  >
                    {sourceFile ? (
                      <div className="flex min-h-[160px] flex-col justify-between rounded-xl bg-white p-4 shadow-sm ring-1 ring-black/5">
                        <div className="flex items-start gap-3">
                          <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-emerald-50 text-emerald-700">
                            <Icon name="document" />
                          </span>
                          <div className="min-w-0">
                            <p className="truncate text-sm font-bold text-[#242236]">
                              {sourceFile.name}
                            </p>
                            <p className="mt-1 text-[11px] text-[#8a8798]">
                              {(sourceFile.size / 1024).toFixed(1)} KB · ready to check
                            </p>
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button
                            type="button"
                            onClick={() => fileInputRef.current?.click()}
                            className="flex-1 rounded-lg border border-[#dfdce7] px-3 py-2 text-[11px] font-bold text-[#625e74] hover:bg-[#f8f7fa]"
                          >
                            Replace
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setSourceFile(null)
                              resetOutput()
                              if (fileInputRef.current) fileInputRef.current.value = ''
                            }}
                            className="flex-1 rounded-lg border border-rose-200 px-3 py-2 text-[11px] font-bold text-rose-600 hover:bg-rose-50"
                          >
                            Remove
                          </button>
                        </div>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="flex min-h-[160px] w-full flex-col items-center justify-center rounded-xl text-center"
                      >
                        <span className="grid h-12 w-12 place-items-center rounded-2xl bg-[#ece9ff] text-[#6548e8]">
                          <Icon name="upload" className="h-6 w-6" />
                        </span>
                        <span className="mt-4 text-sm font-extrabold">
                          Drop a source here
                        </span>
                        <span className="mt-1 text-[11px] text-[#8b879a]">
                          or click to browse · PDF or TXT · max 10 MB
                        </span>
                      </button>
                    )}
                  </div>

                  <div className="my-5 h-px bg-[#e8e5ed]" />

                  <div className="space-y-3">
                    {[
                      'Claim-by-claim verdicts',
                      'Exact supporting passages',
                      'Privacy and publication flags',
                    ].map((feature) => (
                      <div
                        key={feature}
                        className="flex items-center gap-2.5 text-xs font-semibold text-[#676376]"
                      >
                        <span className="grid h-5 w-5 place-items-center rounded-full bg-[#e8f8ee] text-emerald-700">
                          <Icon name="check" className="h-3.5 w-3.5" />
                        </span>
                        {feature}
                      </div>
                    ))}
                  </div>

                  <div className="mt-auto pt-6">
                    <button
                      type="button"
                      disabled={!readyToAudit}
                      onClick={runAudit}
                      className="flex w-full items-center justify-between rounded-xl bg-[#7357ff] px-5 py-4 text-sm font-extrabold text-white shadow-[0_12px_25px_-12px_rgba(93,62,226,0.8)] transition hover:-translate-y-0.5 hover:bg-[#6548e8] disabled:cursor-not-allowed disabled:translate-y-0 disabled:bg-[#cbc8d6] disabled:shadow-none"
                    >
                      <span>
                        {isAuditing ? auditSteps[auditStep] : 'Run evidence audit'}
                      </span>
                      {isAuditing ? (
                        <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                      ) : (
                        <Icon name="arrow" />
                      )}
                    </button>
                    <p className="mt-3 text-center text-[10px] leading-4 text-[#9a96a7]">
                      Don’t upload confidential, regulated, or privileged files.
                    </p>
                  </div>
                </div>
              </div>

              {isAuditing && (
                <div
                  aria-live="polite"
                  className="border-t border-[#ded8ff] bg-[#f1edff] px-5 py-4 sm:px-7"
                >
                  <div className="flex items-center gap-4">
                    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-[#7357ff] text-white">
                      {auditStep + 1}
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex justify-between text-xs font-bold text-[#43329d]">
                        <span>{auditSteps[auditStep]}</span>
                        <span>
                          {auditStep + 1} / {auditSteps.length}
                        </span>
                      </div>
                      <div className="mt-2 h-1 overflow-hidden rounded-full bg-[#d8cffd]">
                        <div
                          className="h-full rounded-full bg-[#7357ff] transition-all duration-700"
                          style={{
                            width: `${((auditStep + 1) / auditSteps.length) * 100}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {error && (
                <div
                  role="alert"
                  className="border-t border-rose-200 bg-rose-50 px-5 py-4 text-center text-sm font-semibold text-rose-800"
                >
                  {error}
                </div>
              )}
            </section>

            {!result && (
              <section className="mt-5 grid gap-3 md:grid-cols-3">
                {[
                  ['01', 'We extract the claims', 'Important factual statements are separated from opinion.'],
                  ['02', 'We find the receipts', 'Each claim is compared with the strongest source passage.'],
                  ['03', 'You get a clear call', 'Supported, contradicted, or flagged for human review.'],
                ].map(([number, title, copy]) => (
                  <article
                    key={number}
                    className="flex gap-4 rounded-2xl border border-[#e1dee8] bg-[#faf9fc] p-4"
                  >
                    <span className="text-xs font-extrabold text-[#7357ff]">
                      {number}
                    </span>
                    <div>
                      <h3 className="text-xs font-extrabold">{title}</h3>
                      <p className="mt-1 text-[11px] leading-5 text-[#817d8f]">
                        {copy}
                      </p>
                    </div>
                  </article>
                ))}
              </section>
            )}

            {result && (
              <section
                id="audit-report"
                className="mt-8 scroll-mt-24 overflow-hidden rounded-[24px] border border-[#dcd9e6] bg-white shadow-[0_20px_60px_-38px_rgba(39,32,72,0.35)]"
              >
                <div
                  className={`px-5 py-7 sm:px-8 ${
                    result.summary.publication_status === 'blocked'
                      ? 'bg-[#fff1f2]'
                      : 'bg-[#ecfdf5]'
                  }`}
                >
                  <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
                    <div className="flex items-start gap-4">
                      <span
                        className={`grid h-12 w-12 shrink-0 place-items-center rounded-2xl text-white ${
                          result.summary.publication_status === 'blocked'
                            ? 'bg-rose-600'
                            : 'bg-emerald-600'
                        }`}
                      >
                        <LogoMark />
                      </span>
                      <div>
                        <p className="text-[10px] font-extrabold uppercase tracking-[0.18em] text-[#777386]">
                          Audit complete
                        </p>
                        <h2 className="mt-1 text-2xl font-extrabold tracking-[-0.04em]">
                          {result.summary.publication_status === 'blocked'
                            ? 'A few claims need attention.'
                            : 'Ready for your final review.'}
                        </h2>
                        <p className="mt-1 text-xs text-[#777386]">
                          {result.audit_id} · {result.source.filename}
                        </p>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={downloadReport}
                      className="rounded-xl border border-[#d9d5e2] bg-white px-4 py-3 text-xs font-bold shadow-sm hover:bg-[#faf9fc]"
                    >
                      Download JSON report
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-px bg-[#e4e1e9] sm:grid-cols-4">
                  {[
                    ['Supported', result.summary.supported, 'text-emerald-600'],
                    [
                      'Needs review',
                      result.summary.weakly_supported +
                        result.summary.human_review_required,
                      'text-violet-600',
                    ],
                    ['Unsupported', result.summary.unsupported, 'text-orange-600'],
                    ['Contradicted', result.summary.contradicted, 'text-rose-600'],
                  ].map(([label, value, color]) => (
                    <div key={label} className="bg-white px-5 py-5 sm:px-7">
                      <p className={`text-3xl font-extrabold ${color}`}>{value}</p>
                      <p className="mt-1 text-[11px] font-bold text-[#817d8f]">
                        {label}
                      </p>
                    </div>
                  ))}
                </div>

                <div className="border-t border-[#e4e1e9] bg-[#f8f7fa] p-5 sm:p-7">
                  <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
                    <div>
                      <p className="text-[10px] font-extrabold uppercase tracking-[0.18em] text-[#7357ff]">
                        Claim review
                      </p>
                      <h3 className="mt-1 text-xl font-extrabold tracking-[-0.03em]">
                        {result.summary.total_claims} claims checked
                      </h3>
                    </div>
                    <p className="text-[11px] font-semibold text-[#817d8f]">
                      {result.source.page_count} page
                      {result.source.page_count === 1 ? '' : 's'} ·{' '}
                      {result.source.chunk_count} evidence passages
                    </p>
                  </div>

                  <div className="grid gap-4 xl:grid-cols-2">
                    {result.claim_audits.map((audit) => {
                      const meta = verdictMeta[audit.verdict]
                      return (
                        <article
                          key={audit.claim.id}
                          className={`rounded-2xl border bg-white p-5 ${meta.border}`}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                              <span className="grid h-7 min-w-7 place-items-center rounded-lg bg-[#f0eef4] px-2 text-[10px] font-extrabold text-[#696577]">
                                {audit.claim.id}
                              </span>
                              <span className="text-[9px] font-extrabold uppercase tracking-[0.14em] text-[#aaa6b5]">
                                {audit.claim.type.replaceAll('_', ' ')}
                              </span>
                            </div>
                            <span
                              className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[10px] font-extrabold ring-1 ${meta.badge}`}
                            >
                              <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
                              {meta.label}
                            </span>
                          </div>
                          <p className="mt-4 text-sm font-extrabold leading-6">
                            {audit.claim.text}
                          </p>
                          <p className="mt-2 text-xs leading-5 text-[#747083]">
                            {audit.explanation}
                          </p>
                          {audit.evidence_quote && (
                            <blockquote className="mt-4 rounded-xl border-l-2 border-[#7357ff] bg-[#f7f5ff] px-4 py-3 text-xs italic leading-5 text-[#5e596d]">
                              “{audit.evidence_quote}”
                            </blockquote>
                          )}
                        </article>
                      )
                    })}
                  </div>
                </div>

                {result.privacy_findings.length > 0 && (
                  <div className="border-t border-amber-200 bg-amber-50 px-5 py-5 sm:px-7">
                    <h3 className="text-sm font-extrabold text-amber-900">
                      Privacy review · {result.privacy_findings.length} finding
                      {result.privacy_findings.length === 1 ? '' : 's'}
                    </h3>
                    <div className="mt-3 grid gap-3 md:grid-cols-2">
                      {result.privacy_findings.map((finding, index) => (
                        <div
                          key={`${finding.kind}-${index}`}
                          className="rounded-xl bg-white p-4 text-xs ring-1 ring-amber-200"
                        >
                          <p className="font-bold">
                            {finding.label}: {finding.masked_value}
                          </p>
                          <p className="mt-1 leading-5 text-[#747083]">
                            {finding.recommendation}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            <footer className="flex flex-col justify-between gap-2 px-1 pb-3 pt-8 text-[10px] font-semibold text-[#9b97a7] sm:flex-row">
              <p>ProofPilot · Evidence before confidence.</p>
              <p>Your content, your sources, your final decision.</p>
            </footer>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
