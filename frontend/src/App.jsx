import React, { useState, useEffect, useRef } from 'react';
import {
  Activity, Sparkles, Send, Database, Network, Layers,
  Clock, FileText, AlertTriangle, Bot, User, Stethoscope,
  Sliders, Info, Sparkle, BarChart3, ChevronDown, ChevronUp,
  RefreshCw, Zap, Archive
} from 'lucide-react';

const STARTER_PROMPTS = [
  { text: "What are the primary symptoms and warning signs of asthma?", icon: "🫁", tag: "Respiratory" },
  { text: "Compare therapeutic uses and toxicity of Aspirin vs Acetaminophen", icon: "💊", tag: "Pharmacology" },
  { text: "What are the clinical signs and diagnostic tests for acute appendicitis?", icon: "🩺", tag: "Surgery & GI" },
  { text: "What are the side effects and contraindications of beta-blockers?", icon: "❤️", tag: "Cardiovascular" },
  { text: "🚨 Emergency: Severe crushing chest pain radiating to left arm", icon: "🚨", tag: "Emergency" }
];

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 'welcome', role: 'assistant', cached: false,
      content: "### Welcome to GaleMed AI Clinical Assistant 🩺\n\nI am an evidence-based clinical intelligence system powered by **The Gale Encyclopedia of Medicine (3rd Edition)** and **Neo4j Hybrid GraphRAG**.\n\n**Core Capabilities:**\n- 🔍 **Hybrid Multi-Vector Search**: Qdrant dense vector + BM25 keyword matching fused with RRF.\n- 🕸️ **Knowledge Graph (GraphRAG)**: Real-time Cypher traversal connecting 288 Diseases, 333 Medications, and 508 Symptoms.\n- ⚡ **Redis Semantic Cache**: Instant answers (< 10ms) for semantically similar questions previously answered.\n- 🌊 **Token Streaming**: Characters appear in real-time as the AI generates them.\n\n*Select a starter question or enter a clinical inquiry to begin.*",
      sources: [], latency: {}
    }
  ]);
  const [input, setInput] = useState('');
  const [searchMode, setSearchMode] = useState('auto');
  const [useGraph, setUseGraph] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState({ vectors: 13350, nodes: 1800, status: 'connecting' });
  const [cacheStats, setCacheStats] = useState({ hits: 0, misses: 0, hit_rate_pct: 0, total_cached_queries: 0, available: false });
  const [selectedMessage, setSelectedMessage] = useState(null);
  const [activeTab, setActiveTab] = useState('sources');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [expandedSources, setExpandedSources] = useState({});
  const messagesEndRef = useRef(null);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/stats');
      if (res.ok) {
        const data = await res.json();
        setStats({
          vectors: data.vector_store?.points_count || 13350,
          nodes: data.graph_counts ? Object.values(data.graph_counts).reduce((a, b) => a + b, 0) : 1800,
          status: 'online'
        });
      }
    } catch (_) { setStats(prev => ({ ...prev, status: 'offline' })); }
  };

  const fetchCacheStats = async () => {
    try {
      const res = await fetch('/api/cache/stats');
      if (res.ok) {
        const data = await res.json();
        setCacheStats(data.cache || {});
      }
    } catch (_) {}
  };

  useEffect(() => { fetchStats(); fetchCacheStats(); }, []);
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isLoading]);

  const handleSend = async (queryText = input) => {
    const q = queryText.trim();
    if (!q || isLoading) return;
    setInput('');

    const userMsg = {
      id: Date.now().toString(), role: 'user', content: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    // Placeholder AI message for streaming
    const aiId = (Date.now() + 1).toString();
    const aiPlaceholder = {
      id: aiId, role: 'assistant', content: '', sources: [], latency: {},
      metadata: {}, cached: false, streaming: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, aiPlaceholder]);

    try {
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, search_mode: searchMode, top_k: 10, use_graph: useGraph })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullAnswer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'token') {
              fullAnswer += data.token;
              setMessages(prev => prev.map(m =>
                m.id === aiId ? { ...m, content: fullAnswer, cached: data.cached || false } : m
              ));
            } else if (data.type === 'done') {
              const finalContent = fullAnswer || data.answer || '';
              setMessages(prev => prev.map(m =>
                m.id === aiId ? {
                  ...m, content: finalContent, streaming: false,
                  sources: data.sources || [], latency: data.latency || {},
                  metadata: data.metadata || {}, cached: data.cached || false,
                  cache_hit_count: data.cache_hit_count
                } : m
              ));
              setSelectedMessage({ sources: data.sources || [], latency: data.latency || {} });
              fetchCacheStats();
            } else if (data.type === 'error') {
              setMessages(prev => prev.map(m =>
                m.id === aiId ? { ...m, content: `⚠️ Error: ${data.message}`, streaming: false, isError: true } : m
              ));
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === aiId ? {
          ...m, content: `⚠️ **Connection Error**: ${err.message}`, streaming: false, isError: true
        } : m
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const renderMarkdown = (text) => {
    if (!text) return null;
    return text.split('\n').map((line, idx) => {
      if (line.includes('🚨') || line.includes('EMERGENCY')) {
        return (
          <div key={idx} className="my-3 p-4 bg-red-950/70 border border-red-500/60 rounded-xl text-red-200 shadow-[0_0_20px_rgba(239,68,68,0.25)] flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 text-red-400 shrink-0 mt-0.5 animate-pulse" />
            <div>
              <div className="font-bold text-red-400 text-sm">CRITICAL MEDICAL ALERT</div>
              <div className="text-xs text-red-200 mt-1 font-medium leading-relaxed">{line.replace(/[*#🚨]/g, '')}</div>
            </div>
          </div>
        );
      }
      if (line.startsWith('### ')) return <h4 key={idx} className="text-cyan-400 font-bold text-base mt-4 mb-2 flex items-center gap-2"><Sparkle className="w-3.5 h-3.5" />{line.replace('### ', '')}</h4>;
      if (line.startsWith('## ')) return <h3 key={idx} className="text-cyan-300 font-extrabold text-lg mt-5 mb-2 pb-1 border-b border-white/10">{line.replace('## ', '')}</h3>;
      if (line.startsWith('# ')) return <h2 key={idx} className="text-white font-extrabold text-xl mt-6 mb-3">{line.replace('# ', '')}</h2>;
      if (line.startsWith('- ') || line.startsWith('* ')) {
        return <li key={idx} className="ml-4 my-1 text-slate-300 list-disc text-sm leading-relaxed">{formatInlineText(line.substring(2))}</li>;
      }
      if (line.includes('educational purposes only') || line.includes('not constitute medical advice')) {
        return (
          <div key={idx} className="mt-4 pt-3 border-t border-white/10 text-xs text-slate-400 italic flex items-center gap-2 bg-white/[0.02] p-2.5 rounded-lg">
            <Info className="w-4 h-4 text-amber-400 shrink-0" />
            <span>{line.replace(/[*_]/g, '')}</span>
          </div>
        );
      }
      if (!line.trim()) return <div key={idx} className="h-2" />;
      return <p key={idx} className="text-slate-200 text-sm leading-relaxed my-1.5">{formatInlineText(line)}</p>;
    });
  };

  const formatInlineText = (text) => {
    return text.split(/(\[\d+\]|\*\*.*?\*\*)/g).map((part, i) => {
      if (/^\[\d+\]$/.test(part)) {
        return <span key={i} onClick={() => setActiveTab('sources')} className="inline-flex items-center px-1.5 py-0.5 mx-0.5 rounded text-[11px] font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-500/40 cursor-pointer hover:bg-cyan-900 transition-colors" title="Click to view source">{part}</span>;
      }
      if (part.startsWith('**') && part.endsWith('**')) return <strong key={i} className="text-white font-semibold">{part.slice(2, -2)}</strong>;
      return part;
    });
  };

  const currentSources = selectedMessage?.sources || messages.filter(m => m.sources?.length > 0).slice(-1)[0]?.sources || [];
  const currentLatency = selectedMessage?.latency || messages.filter(m => m.latency?.total).slice(-1)[0]?.latency || {};

  return (
    <div className="flex h-screen w-full bg-[#07090E] text-slate-100 overflow-hidden" style={{ fontFamily: "'Plus Jakarta Sans', 'Inter', sans-serif" }}>

      {/* LEFT SIDEBAR */}
      <aside className="w-72 bg-[#0B0F19]/90 border-r border-white/10 flex flex-col justify-between shrink-0 p-4 backdrop-blur-xl">
        <div className="space-y-5">
          {/* Brand */}
          <div className="flex items-center gap-3 px-2 py-1">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.4)]">
              <Stethoscope className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="font-extrabold text-base tracking-tight text-white flex items-center gap-1.5">
                GaleMed AI
                <span className="text-[10px] font-mono bg-cyan-950/80 text-cyan-400 border border-cyan-500/30 px-1.5 py-0.5 rounded">v2.0</span>
              </div>
              <div className="text-[11px] text-slate-400">Clinical GraphRAG + Redis Cache</div>
            </div>
          </div>

          {/* DB Stats */}
          <div className="p-3 bg-white/[0.03] border border-white/10 rounded-xl space-y-2.5">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center justify-between">
              <span>Knowledge Base</span>
              <span className="flex items-center gap-1 text-emerald-400 text-[10px] font-semibold">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" /> Live
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                { icon: Database, label: "Qdrant", value: stats.vectors.toLocaleString(), sub: "Vectors", color: "text-cyan-400" },
                { icon: Network, label: "Neo4j", value: stats.nodes.toLocaleString(), sub: "Entities", color: "text-teal-400" }
              ].map(({ icon: Icon, label, value, sub, color }) => (
                <div key={label} className="bg-black/40 p-2 rounded-lg border border-white/5">
                  <div className={"flex items-center gap-1.5 font-medium text-[11px] " + color}>
                    <Icon className="w-3.5 h-3.5" /> {label}
                  </div>
                  <div className="font-mono font-bold text-white text-sm mt-0.5">{value}</div>
                  <div className="text-[10px] text-slate-500">{sub}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Redis Cache Stats */}
          <div className="p-3 bg-white/[0.03] border border-white/10 rounded-xl space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center justify-between">
              <div className="flex items-center gap-1.5"><Archive className="w-3 h-3" /> Redis Cache</div>
              <span className={"text-[10px] font-semibold " + (cacheStats.available ? "text-emerald-400" : "text-red-400")}>
                {cacheStats.available ? "● Online" : "○ Offline"}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1.5 text-[11px]">
              <div className="bg-black/30 rounded-lg p-1.5 text-center">
                <div className="font-mono font-bold text-green-400">{cacheStats.hits || 0}</div>
                <div className="text-slate-500 text-[10px]">Hits</div>
              </div>
              <div className="bg-black/30 rounded-lg p-1.5 text-center">
                <div className="font-mono font-bold text-slate-300">{cacheStats.misses || 0}</div>
                <div className="text-slate-500 text-[10px]">Misses</div>
              </div>
              <div className="bg-black/30 rounded-lg p-1.5 text-center">
                <div className="font-mono font-bold text-cyan-300">{cacheStats.hit_rate_pct || 0}%</div>
                <div className="text-slate-500 text-[10px]">Hit Rate</div>
              </div>
            </div>
            <div className="text-[10px] text-slate-400 flex items-center justify-between px-0.5">
              <span>{cacheStats.total_cached_queries || 0} queries cached</span>
              <button onClick={fetchCacheStats} className="hover:text-cyan-400 transition-colors"><RefreshCw className="w-2.5 h-2.5" /></button>
            </div>
          </div>

          {/* Search Mode */}
          <div className="space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 flex items-center gap-1.5 px-1">
              <Sliders className="w-3.5 h-3.5" /> Search Strategy
            </div>
            <div className="grid grid-cols-2 gap-1.5 bg-black/40 p-1.5 rounded-xl border border-white/5 text-xs">
              {[{ id: 'auto', label: '⚡ Auto' }, { id: 'hybrid', label: '🔀 Hybrid' }, { id: 'vector', label: '🧠 Vector' }, { id: 'graph', label: '🕸️ Graph' }].map(mode => (
                <button key={mode.id} onClick={() => setSearchMode(mode.id)} className={"py-1.5 px-2 rounded-lg transition-all text-center font-medium " + (searchMode === mode.id ? "bg-gradient-to-r from-cyan-600 to-teal-600 text-white font-bold shadow-[0_0_12px_rgba(6,182,212,0.3)]" : "text-slate-400 hover:text-white hover:bg-white/5")}>
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          {/* Starters */}
          <div className="space-y-2">
            <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 px-1">Clinical Examples</div>
            <div className="space-y-1.5">
              {STARTER_PROMPTS.map((p, idx) => (
                <button key={idx} onClick={() => handleSend(p.text)} className="w-full text-left p-2.5 rounded-xl bg-white/[0.02] border border-white/5 hover:border-cyan-500/40 hover:bg-cyan-950/20 text-xs text-slate-300 transition-all flex items-start gap-2 group">
                  <span className="shrink-0 mt-0.5">{p.icon}</span>
                  <div className="flex-1 line-clamp-2 leading-tight group-hover:text-cyan-200">{p.text}</div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-white/10 text-[11px] text-slate-500 flex justify-between">
          <span>The Gale Encyclopedia</span>
          <button onClick={() => setMessages([{ id: '1', role: 'assistant', content: 'Chat cleared. How may I assist?', sources: [], latency: {}, cached: false }])} className="hover:text-cyan-400 flex items-center gap-1 transition-colors">
            <RefreshCw className="w-3 h-3" /> Clear
          </button>
        </div>
      </aside>

      {/* CENTER CHAT */}
      <main className="flex-1 flex flex-col bg-[#07090E] overflow-hidden">
        <header className="h-14 border-b border-white/10 bg-[#0B0F19]/80 backdrop-blur-md px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3 text-xs font-mono text-slate-300">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            Mode: <strong className="text-cyan-400 uppercase">{searchMode}</strong>
            <span className="text-slate-600">•</span>
            Graph: <strong className="text-teal-400">{useGraph ? 'ON' : 'OFF'}</strong>
            <span className="text-slate-600">•</span>
            Cache: <strong className={cacheStats.available ? "text-green-400" : "text-red-400"}>{cacheStats.available ? 'Redis ●' : 'Offline ○'}</strong>
          </div>
          <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="text-xs px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-slate-300 flex items-center gap-1.5 transition-all">
            <BarChart3 className="w-3.5 h-3.5 text-cyan-400" /> {isSidebarOpen ? 'Hide' : 'Show'} Telemetry
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((msg) => (
            <div key={msg.id} onClick={() => msg.sources?.length > 0 && setSelectedMessage(msg)} className={"flex gap-3 " + (msg.role === 'user' ? 'justify-end' : 'justify-start') + " animate-[fadeIn_0.25s_ease-out_forwards]"}>
              {msg.role === 'assistant' && (
                <div className={"w-8 h-8 rounded-xl bg-gradient-to-br flex items-center justify-center text-white shrink-0 mt-1 " + (msg.cached ? "from-green-500 to-emerald-600 shadow-[0_0_15px_rgba(34,197,94,0.3)]" : "from-cyan-500 to-teal-600 shadow-[0_0_15px_rgba(6,182,212,0.3)]")}>
                  {msg.cached ? <Zap className="w-4 h-4" /> : <Bot className="w-5 h-5" />}
                </div>
              )}

              <div className={"max-w-3xl rounded-2xl p-5 shadow-2xl transition-all " + (msg.role === 'user' ? "bg-gradient-to-br from-cyan-950/60 to-slate-900 border border-cyan-500/30 ml-12" : "bg-[rgba(14,19,31,0.75)] backdrop-blur-md border border-white/10 mr-12")}>
                {msg.role === 'assistant' && (
                  <div className="flex items-center justify-between text-xs font-mono pb-3 border-b border-white/10 mb-3">
                    <div className="flex items-center gap-2 font-semibold">
                      {msg.cached
                        ? <span className="flex items-center gap-1.5 text-green-400"><Zap className="w-3.5 h-3.5" /> Cache Hit · Redis</span>
                        : <span className="flex items-center gap-1.5 text-cyan-400"><Sparkles className="w-3.5 h-3.5" /> GaleMed Synthesizer</span>
                      }
                    </div>
                    <div className="flex items-center gap-2 text-slate-400">
                      {msg.cached && <span className="px-2 py-0.5 rounded bg-green-950 border border-green-500/40 text-green-300 text-[11px]">⚡ {msg.latency?.cache_lookup_ms ? msg.latency.cache_lookup_ms.toFixed(0) + 'ms' : '< 10ms'}</span>}
                      {!msg.cached && msg.latency?.total && <span className="px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500/30 text-cyan-300 text-[11px]">⚡ {(msg.latency.total / 1000).toFixed(2)}s</span>}
                      {msg.streaming && <span className="text-cyan-400 text-[11px] flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" /> streaming</span>}
                    </div>
                  </div>
                )}
                <div className="space-y-1 text-sm leading-relaxed">{renderMarkdown(msg.content)}</div>
                {msg.streaming && <span className="inline-block w-1.5 h-4 bg-cyan-400 ml-0.5 animate-pulse rounded-sm" />}
              </div>

              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-xl bg-slate-800 border border-white/10 flex items-center justify-center shrink-0 mt-1">
                  <User className="w-4 h-4 text-slate-300" />
                </div>
              )}
            </div>
          ))}

          {isLoading && !messages.some(m => m.streaming) && (
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 flex items-center justify-center animate-pulse">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-[rgba(14,19,31,0.75)] border border-white/10 rounded-2xl p-4 flex items-center gap-3">
                <div className="flex space-x-1.5">
                  {[0, 150, 300].map(d => <span key={d} className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-bounce" style={{ animationDelay: d + 'ms' }} />)}
                </div>
                <span className="text-xs font-mono text-cyan-300">Querying Qdrant & Neo4j Cypher Graph...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="p-4 bg-gradient-to-t from-[#07090E] via-[#07090E]/90 to-transparent">
          <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="max-w-4xl mx-auto flex items-center bg-[#0F1422] border border-white/15 focus-within:border-cyan-500/60 rounded-2xl p-2 shadow-[0_4px_30px_rgba(0,0,0,0.6)] backdrop-blur-xl transition-all">
            <input type="text" value={input} onChange={e => setInput(e.target.value)} placeholder="Ask GaleMed about medical symptoms, medications, or query the clinical graph..." className="flex-1 bg-transparent px-4 py-2.5 text-sm text-white placeholder:text-slate-500 focus:outline-none" disabled={isLoading} />
            <button type="submit" disabled={!input.trim() || isLoading} className="w-10 h-10 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center text-white shadow-[0_0_15px_rgba(6,182,212,0.4)] transition-all">
              <Send className="w-4 h-4" />
            </button>
          </form>
          <div className="text-center text-[10px] text-slate-500 mt-2">GaleMed AI · Hybrid GraphRAG + Redis Semantic Cache · Educational guidance only.</div>
        </div>
      </main>

      {/* RIGHT TELEMETRY */}
      {isSidebarOpen && (
        <aside className="w-80 bg-[#0B0F19]/90 border-l border-white/10 flex flex-col shrink-0 backdrop-blur-xl">
          <div className="flex border-b border-white/10 p-2 gap-1 text-xs font-medium">
            {[{ id: 'sources', icon: FileText, label: `Evidence (${currentSources.length})` }, { id: 'telemetry', icon: Clock, label: 'Latency' }].map(({ id, icon: Icon, label }) => (
              <button key={id} onClick={() => setActiveTab(id)} className={"flex-1 py-2 rounded-lg flex items-center justify-center gap-1.5 transition-all " + (activeTab === id ? "bg-cyan-950/80 text-cyan-300 border border-cyan-500/30 font-bold" : "text-slate-400 hover:text-white")}>
                <Icon className="w-3.5 h-3.5" /> {label}
              </button>
            ))}
          </div>

          {activeTab === 'sources' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {currentSources.length === 0 ? (
                <div className="text-center py-12 text-slate-500 text-xs"><Layers className="w-8 h-8 mx-auto mb-2 opacity-30" /> No sources retrieved yet.</div>
              ) : (
                currentSources.map((s, idx) => (
                  <div key={idx} className="bg-[rgba(20,27,45,0.6)] border border-white/6 rounded-xl p-3 space-y-2 hover:border-cyan-500/30 transition-all">
                    <div className="flex items-start justify-between gap-2">
                      <div className="font-semibold text-xs text-cyan-300 line-clamp-1"><span className="font-mono text-slate-400">[{idx + 1}]</span> {s.doc_id}</div>
                      <span className={"text-[10px] font-mono px-1.5 py-0.5 rounded font-bold " + (s.source === 'graph' ? "bg-teal-950 text-teal-300 border border-teal-500/40" : "bg-cyan-950 text-cyan-300 border border-cyan-500/40")}>{s.source}</span>
                    </div>
                    <div className="text-[11px] text-slate-300 font-mono bg-black/40 p-2 rounded border border-white/5 leading-relaxed">
                      {expandedSources[idx] ? s.content_preview : s.content_preview?.slice(0, 120) + '...'}
                    </div>
                    <button onClick={() => setExpandedSources(p => ({ ...p, [idx]: !p[idx] }))} className="text-[10px] text-cyan-400 flex items-center gap-1">
                      {expandedSources[idx] ? <><ChevronUp className="w-3 h-3" /> Less</> : <><ChevronDown className="w-3 h-3" /> Full Passage</>}
                    </button>
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === 'telemetry' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <div className="p-3 bg-black/40 rounded-xl border border-white/5 text-center">
                <div className="text-[11px] font-mono text-slate-400">Total Execution Time</div>
                <div className="text-2xl font-mono font-bold text-cyan-400 mt-1">
                  {currentLatency.total ? (currentLatency.total > 100 ? (currentLatency.total / 1000).toFixed(3) + 's' : currentLatency.total.toFixed(1) + 'ms') : '—'}
                </div>
                {currentLatency.cache_lookup_ms && <div className="text-[10px] text-green-400 mt-0.5">⚡ Cache Hit · Redis</div>}
              </div>
              <div className="space-y-3 text-xs">
                {[
                  { label: 'Query Classification', val: currentLatency.query_classify, color: 'bg-indigo-500' },
                  { label: 'Hybrid Retrieval', val: currentLatency.retrieval, color: 'bg-cyan-500' },
                  { label: 'Neo4j Graph Cypher', val: currentLatency.graph_search, color: 'bg-teal-500' },
                  { label: 'Rerank & MMR', val: currentLatency.post_retrieval, color: 'bg-amber-500' },
                  { label: 'LLM Generation', val: currentLatency.generation, color: 'bg-emerald-500' },
                ].map(({ label, val, color }) => val ? (
                  <div key={label} className="space-y-1">
                    <div className="flex justify-between text-[11px] text-slate-300 font-mono">
                      <span>{label}</span>
                      <span className="font-bold">{(val * 1000).toFixed(0)}ms</span>
                    </div>
                    <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                      <div className={"h-full rounded-full " + color} style={{ width: `${Math.min(100, Math.max(5, (val / (currentLatency.total / 1000 || 5)) * 100))}%` }} />
                    </div>
                  </div>
                ) : null)}
              </div>
            </div>
          )}
        </aside>
      )}
    </div>
  );
}
