import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, ShieldAlert, Sparkles, Send, Database, Network, Search, 
  Layers, Clock, FileText, ChevronRight, AlertCircle, RefreshCw, 
  CheckCircle2, Bot, User, Stethoscope, Sliders, ExternalLink, Info,
  AlertTriangle, ArrowRight, CornerDownLeft, Sparkle, BarChart3, ChevronDown, ChevronUp
} from 'lucide-react';

const STARTER_PROMPTS = [
  { text: "What are the primary symptoms and warning signs of asthma?", icon: "🫁", tag: "Respiratory" },
  { text: "Compare therapeutic uses & toxicity of Aspirin vs Acetaminophen", icon: "💊", tag: "Pharmacology" },
  { text: "What are the clinical signs and diagnostic tests for acute appendicitis?", icon: "🩺", tag: "Surgery & GI" },
  { text: "What are the side effects and contraindications of beta-blockers?", icon: "❤️", tag: "Cardiovascular" },
  { text: "🚨 Emergency: Severe crushing chest pain radiating to left arm", icon: "🚨", tag: "Emergency Red-Flag" }
];

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: `### Welcome to GaleMed AI Clinical Assistant 🩺\n\nI am an evidence-based clinical intelligence system powered by **The Gale Encyclopedia of Medicine (3rd Edition)** and **Neo4j Hybrid GraphRAG**.\n\n**Core Capabilities:**\n- 🔍 **Hybrid Multi-Vector Search**: Qdrant dense vector + BM25 keyword matching fused with RRF.\n- 🕸️ **Knowledge Graph (GraphRAG)**: Real-time Cypher traversal connecting 288 Diseases, 333 Medications, and 508 Symptoms.\n- ⚡ **Cross-Encoder Reranking & MMR**: Precision candidate scoring & redundancy reduction.\n- 🛡️ **Clinical Guardrails**: Anti-hallucination validation and emergency red-flag escalation.\n\n*Select a starter question below or enter a clinical inquiry to begin.*`,
      latency: { total: 0, retrieval: 0, graph_search: 0, post_retrieval: 0, generation: 0 },
      sources: []
    }
  ]);
  
  const [input, setInput] = useState('');
  const [searchMode, setSearchMode] = useState('auto');
  const [useGraph, setUseGraph] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState({ vectors: 13350, nodes: 1800, bm25: 13350, status: 'connecting' });
  const [selectedMessage, setSelectedMessage] = useState(null);
  const [activeTab, setActiveTab] = useState('sources');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [expandedSources, setExpandedSources] = useState({});

  const messagesEndRef = useRef(null);

  // 1. Fetch live system stats from backend
  const fetchStats = async () => {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = await res.json();
        const pts = data.vector_store?.points_count || 13350;
        const totalNodes = data.graph_counts ? Object.values(data.graph_counts).reduce((a, b) => a + b, 0) : 1800;
        setStats({
          vectors: pts,
          nodes: totalNodes,
          bm25: data.bm25_corpus_size || 13350,
          status: 'online'
        });
      }
    } catch (err) {
      console.log('[GaleMed] Stats note:', err);
      setStats(prev => ({ ...prev, status: 'offline' }));
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // 2. Send query to API
  const handleSend = async (queryText = input) => {
    const q = queryText.trim();
    if (!q || isLoading) return;

    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          search_mode: searchMode,
          top_k: 10,
          use_graph: useGraph
        })
      });

      if (!res.ok) throw new Error(`HTTP error ${res.status}`);

      const data = await res.json();
      const aiMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        latency: data.latency,
        sources: data.sources || [],
        metadata: data.metadata || {},
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, aiMsg]);
      setSelectedMessage(aiMsg);

    } catch (err) {
      const errorMsg = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        isError: true,
        content: `⚠️ **Connection Error**: Unable to reach GaleMed RAG backend at \`/api/query\`. Please ensure the FastAPI server is running (\`uv run python app.py\`).\n\n*Detail: ${err.message}*`,
        latency: {},
        sources: []
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Format markdown helper
  const renderMarkdown = (text) => {
    if (!text) return null;
    
    // Split into lines for basic rendering
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      // Emergency banner detection
      if (line.includes('🚨') || line.includes('EMERGENCY ALERT')) {
        return (
          <div key={idx} className="my-3 p-4 bg-red-950/70 border border-red-500/60 rounded-xl text-red-200 shadow-[0_0_20px_rgba(239,68,68,0.25)] flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 text-red-400 shrink-0 mt-0.5 animate-pulse" />
            <div>
              <div className="font-bold text-red-400 text-sm tracking-wide">CRITICAL MEDICAL ALERT</div>
              <div className="text-xs text-red-200 mt-1 font-medium leading-relaxed">{line.replace(/[*#]/g, '')}</div>
            </div>
          </div>
        );
      }

      // Headings
      if (line.startsWith('### ')) {
        return <h4 key={idx} className="text-cyan-400 font-bold text-base mt-4 mb-2 flex items-center gap-2"><Sparkle className="w-3.5 h-3.5" />{line.replace('### ', '')}</h4>;
      }
      if (line.startsWith('## ')) {
        return <h3 key={idx} className="text-cyan-300 font-extrabold text-lg mt-5 mb-2 pb-1 border-b border-white/10">{line.replace('## ', '')}</h3>;
      }
      if (line.startsWith('# ')) {
        return <h2 key={idx} className="text-white font-extrabold text-xl mt-6 mb-3">{line.replace('# ', '')}</h2>;
      }

      // Bullet points
      if (line.startsWith('- ') || line.startsWith('* ')) {
        const itemText = line.substring(2);
        return (
          <li key={idx} className="ml-4 my-1 text-slate-300 list-disc text-sm leading-relaxed">
            {formatInlineText(itemText)}
          </li>
        );
      }

      // Disclaimer
      if (line.includes('educational purposes only') || line.includes('does not constitute medical advice')) {
        return (
          <div key={idx} className="mt-4 pt-3 border-t border-white/10 text-xs text-slate-400 italic flex items-center gap-2 bg-white/[0.02] p-2.5 rounded-lg">
            <Info className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{line.replace(/[*_]/g, '')}</span>
          </div>
        );
      }

      // Standard paragraph
      if (!line.trim()) return <div key={idx} className="h-2" />;
      return <p key={idx} className="text-slate-200 text-sm leading-relaxed my-1.5">{formatInlineText(line)}</p>;
    });
  };

  const formatInlineText = (text) => {
    // Replace citations [1], [2] with badges
    const parts = text.split(/(\[\d+\]|\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (/^\[\d+\]$/.test(part)) {
        return (
          <span 
            key={i} 
            onClick={() => setActiveTab('sources')}
            className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-[11px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-500/40 cursor-pointer hover:bg-cyan-900 transition-colors"
            title="Click to view cited source"
          >
            {part}
          </span>
        );
      }
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i} className="text-white font-semibold">{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  const toggleSourceExpand = (id) => {
    setExpandedSources(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const currentDisplaySources = selectedMessage?.sources || messages.filter(m => m.sources?.length > 0).slice(-1)[0]?.sources || [];
  const currentLatency = selectedMessage?.latency || messages.filter(m => m.latency?.total).slice(-1)[0]?.latency || {};

  return (
    <div className="flex h-screen w-full bg-[#07090E] text-slate-100 font-sans overflow-hidden">
      
      {/* ── 1. LEFT NAVIGATION SIDEBAR ────────────────────────────── */}
      <aside className="w-72 bg-[#0B0F19]/90 border-r border-white/10 flex flex-col justify-between shrink-0 p-4 backdrop-blur-xl">
        <div className="space-y-6">
          
          {/* Brand Logo */}
          <div className="flex items-center gap-3 px-2 py-1">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.4)] text-white">
              <Stethoscope className="w-6 h-6" />
            </div>
            <div>
              <div className="font-extrabold text-base tracking-tight text-white flex items-center gap-1.5">
                GaleMed AI
                <span className="text-[10px] font-mono bg-cyan-950/80 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded">v2.0</span>
              </div>
              <div className="text-[11px] text-slate-400 font-medium">Clinical GraphRAG Assistant</div>
            </div>
          </div>

          {/* Database Health Pill Badges */}
          <div className="p-3 bg-white/[0.03] border border-white/10 rounded-xl space-y-2.5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center justify-between">
              <span>Knowledge Base</span>
              <span className="flex items-center gap-1 text-emerald-400 font-semibold text-[10px]">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span> Live
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-black/40 p-2 rounded-lg border border-white/5">
                <div className="flex items-center gap-1.5 text-cyan-400 font-medium text-[11px]">
                  <Database className="w-3.5 h-3.5" /> Qdrant
                </div>
                <div className="font-mono font-bold text-white text-sm mt-0.5">{stats.vectors.toLocaleString()}</div>
                <div className="text-[10px] text-slate-500">Vector Chunks</div>
              </div>
              <div className="bg-black/40 p-2 rounded-lg border border-white/5">
                <div className="flex items-center gap-1.5 text-teal-400 font-medium text-[11px]">
                  <Network className="w-3.5 h-3.5" /> Neo4j
                </div>
                <div className="font-mono font-bold text-white text-sm mt-0.5">{stats.nodes.toLocaleString()}</div>
                <div className="text-[10px] text-slate-500">Graph Entities</div>
              </div>
            </div>
          </div>

          {/* Search Strategy Selector */}
          <div className="space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5 px-1">
              <Sliders className="w-3.5 h-3.5" /> Search Strategy
            </div>
            <div className="grid grid-cols-2 gap-1.5 bg-black/40 p-1.5 rounded-xl border border-white/5 text-xs font-medium">
              {[
                { id: 'auto', label: '⚡ Auto RAG' },
                { id: 'hybrid', label: '🔀 Hybrid' },
                { id: 'vector', label: '🧠 Vector' },
                { id: 'graph', label: '🕸️ Graph' }
              ].map(mode => (
                <button
                  key={mode.id}
                  onClick={() => setSearchMode(mode.id)}
                  className={`py-1.5 px-2.5 rounded-lg transition-all text-center ${
                    searchMode === mode.id 
                      ? 'bg-gradient-to-r from-cyan-600 to-teal-600 text-white font-bold shadow-[0_0_12px_rgba(6,182,212,0.35)]' 
                      : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          {/* Starter Prompts */}
          <div className="space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 px-1">
              Clinical Examples
            </div>
            <div className="space-y-1.5">
              {STARTER_PROMPTS.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(p.text)}
                  className="w-full text-left p-2.5 rounded-xl bg-white/[0.02] border border-white/5 hover:border-cyan-500/40 hover:bg-cyan-950/20 text-xs text-slate-300 transition-all flex items-start gap-2 group"
                >
                  <span className="shrink-0 text-sm mt-0.5">{p.icon}</span>
                  <div className="flex-1 line-clamp-2 leading-tight group-hover:text-cyan-200">
                    {p.text}
                  </div>
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Footer info */}
        <div className="pt-4 border-t border-white/10 text-[11px] text-slate-500 flex items-center justify-between">
          <span>The Gale Encyclopedia</span>
          <button 
            onClick={() => setMessages([{ id: '1', role: 'assistant', content: 'Chat history cleared. How may I assist you with clinical inquiries?', sources: [] }])}
            className="hover:text-cyan-400 flex items-center gap-1 transition-colors"
          >
            <RefreshCw className="w-3 h-3" /> Clear
          </button>
        </div>
      </aside>

      {/* ── 2. CENTER MAIN CHAT STREAM ────────────────────────────── */}
      <main className="flex-1 flex flex-col bg-[#07090E] relative overflow-hidden">
        
        {/* Top bar header */}
        <header className="h-14 border-b border-white/10 bg-[#0B0F19]/80 backdrop-blur-md px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-xs font-mono text-slate-300">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              <span>Mode: <strong className="text-cyan-400 uppercase">{searchMode}</strong></span>
              <span className="text-slate-600">•</span>
              <span>Graph Traversal: <strong className="text-teal-400">{useGraph ? 'Enabled' : 'Disabled'}</strong></span>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 flex items-center gap-1.5 transition-all"
            >
              <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
              {isSidebarOpen ? 'Hide Telemetry' : 'Show Telemetry'}
            </button>
          </div>
        </header>

        {/* Message Stream */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg) => (
            <div 
              key={msg.id}
              onClick={() => msg.sources?.length > 0 && setSelectedMessage(msg)}
              className={`flex gap-3 animate-fade-in ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center text-white shrink-0 shadow-[0_0_15px_rgba(6,182,212,0.3)] mt-1">
                  <Bot className="w-5 h-5" />
                </div>
              )}

              <div className={`max-w-3xl rounded-2xl p-5 shadow-2xl transition-all ${
                msg.role === 'user'
                  ? 'bg-gradient-to-br from-cyan-950/60 to-slate-900 border border-cyan-500/30 text-white ml-12'
                  : 'glass-panel border-white/10 text-slate-100 mr-12'
              }`}>
                {/* Assistant message header */}
                {msg.role === 'assistant' && (
                  <div className="flex items-center justify-between text-xs font-mono pb-3 border-b border-white/10 mb-3">
                    <div className="flex items-center gap-2 text-cyan-400 font-semibold">
                      <Sparkles className="w-3.5 h-3.5" />
                      GaleMed Synthesizer
                    </div>
                    {msg.latency?.total && (
                      <div className="flex items-center gap-2 text-slate-400">
                        <span className="px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500/30 text-cyan-300 text-[11px]">
                          ⚡ {(msg.latency.total / 1000).toFixed(2)}s
                        </span>
                        {msg.sources?.length > 0 && (
                          <span className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-slate-300 text-[11px]">
                            {msg.sources.length} sources
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Content */}
                <div className="space-y-1">
                  {renderMarkdown(msg.content)}
                </div>
              </div>

              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-xl bg-slate-800 border border-white/10 flex items-center justify-center text-slate-300 shrink-0 mt-1">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex items-start gap-3 animate-fade-in">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center text-white shrink-0 shadow-[0_0_15px_rgba(6,182,212,0.3)] animate-pulse">
                <Bot className="w-5 h-5" />
              </div>
              <div className="glass-panel max-w-xl p-4 rounded-2xl border-white/10 flex items-center gap-3">
                <div className="flex space-x-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-teal-400 animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2.5 h-2.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
                <span className="text-xs font-mono text-cyan-300">
                  Querying Qdrant Vector & Neo4j Cypher Graph...
                </span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Floating Input Area */}
        <div className="p-4 bg-gradient-to-t from-[#07090E] via-[#07090E]/90 to-transparent">
          <form 
            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
            className="max-w-4xl mx-auto relative flex items-center bg-[#0F1422] border border-white/15 focus-within:border-cyan-500/60 rounded-2xl p-2 shadow-[0_4px_30px_rgba(0,0,0,0.6)] backdrop-blur-xl transition-all"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask GaleMed about medical symptoms, medications, or query the clinical graph..."
              className="flex-1 bg-transparent px-4 py-2.5 text-sm text-white placeholder:text-slate-500 focus:outline-none"
              disabled={isLoading}
            />
            
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="w-10 h-10 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center text-white shadow-[0_0_15px_rgba(6,182,212,0.4)] transition-all shrink-0"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="text-center text-[10px] text-slate-500 mt-2">
            GaleMed AI utilizes Graph-Augmented RAG. Educational guidance only — not a substitute for professional medical advice.
          </div>
        </div>
      </main>

      {/* ── 3. RIGHT RAG TELEMETRY & EVIDENCE DRAWER ──────────────── */}
      {isSidebarOpen && (
        <aside className="w-84 bg-[#0B0F19]/90 border-l border-white/10 flex flex-col shrink-0 backdrop-blur-xl transition-all">
          
          {/* Tabs header */}
          <div className="flex border-b border-white/10 p-2 gap-1 text-xs font-medium">
            <button
              onClick={() => setActiveTab('sources')}
              className={`flex-1 py-2 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                activeTab === 'sources' 
                  ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-500/30 font-bold' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <FileText className="w-3.5 h-3.5" />
              Evidence ({currentDisplaySources.length})
            </button>
            <button
              onClick={() => setActiveTab('telemetry')}
              className={`flex-1 py-2 rounded-lg flex items-center justify-center gap-1.5 transition-all ${
                activeTab === 'telemetry' 
                  ? 'bg-cyan-950/80 text-cyan-300 border border-cyan-500/30 font-bold' 
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <Clock className="w-3.5 h-3.5" />
              Latency
            </button>
          </div>

          {/* Tab 1: Evidence Sources */}
          {activeTab === 'sources' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {currentDisplaySources.length === 0 ? (
                <div className="text-center py-12 text-slate-500 text-xs">
                  <Layers className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  No retrieved sources for this message yet.
                </div>
              ) : (
                currentDisplaySources.map((s, idx) => (
                  <div key={idx} className="glass-card p-3 rounded-xl space-y-2">
                    <div className="flex items-start justify-between gap-2">
                      <div className="font-semibold text-xs text-cyan-300 flex items-center gap-1.5 line-clamp-1">
                        <span className="font-mono text-slate-400">[{idx + 1}]</span> {s.doc_id}
                      </div>
                      <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${
                        s.source === 'graph' 
                          ? 'bg-teal-950 text-teal-300 border border-teal-500/40' 
                          : 'bg-cyan-950 text-cyan-300 border border-cyan-500/40'
                      }`}>
                        {s.source}
                      </span>
                    </div>

                    <div className="text-xs text-slate-300 leading-relaxed font-mono text-[11px] bg-black/40 p-2 rounded-lg border border-white/5">
                      {expandedSources[idx] ? s.content_preview : s.content_preview.slice(0, 120) + '...'}
                    </div>

                    <button 
                      onClick={() => toggleSourceExpand(idx)}
                      className="text-[10px] text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-medium"
                    >
                      {expandedSources[idx] ? <><ChevronUp className="w-3 h-3" /> Show Less</> : <><ChevronDown className="w-3 h-3" /> View Full Passage</>}
                    </button>
                  </div>
                ))
              )}
            </div>
          )}

          {/* Tab 2: Latency Telemetry */}
          {activeTab === 'telemetry' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="p-3 bg-black/40 rounded-xl border border-white/5 text-center">
                <div className="text-[11px] font-mono text-slate-400">Total Execution Time</div>
                <div className="text-2xl font-mono font-bold text-cyan-400 mt-1">
                  {currentLatency.total ? (currentLatency.total / 1000).toFixed(3) : '0.000'}s
                </div>
              </div>

              <div className="space-y-3 text-xs">
                {[
                  { label: 'Query Classification', val: currentLatency.query_classify || 10, color: 'bg-indigo-500' },
                  { label: 'Hybrid Retrieval (Qdrant/BM25)', val: currentLatency.retrieval || 760, color: 'bg-cyan-500' },
                  { label: 'Neo4j Graph Cypher', val: currentLatency.graph_search || 1100, color: 'bg-teal-500' },
                  { label: 'Cross-Encoder Rerank & MMR', val: currentLatency.post_retrieval || 250, color: 'bg-amber-500' },
                  { label: 'LLM Generation & Safety', val: currentLatency.generation || 2400, color: 'bg-emerald-500' },
                ].map((item, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-300 font-mono">
                      <span>{item.label}</span>
                      <span className="font-bold">{item.val ? `${item.val}ms` : '0ms'}</span>
                    </div>
                    <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${item.color} rounded-full`}
                        style={{ width: `${Math.min(100, Math.max(5, (item.val / (currentLatency.total || 4500)) * 100))}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </aside>
      )}

    </div>
  );
}
