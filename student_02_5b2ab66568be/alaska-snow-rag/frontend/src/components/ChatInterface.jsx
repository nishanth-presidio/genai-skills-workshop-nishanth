import React, { useState, useEffect } from "react";

const AlaskaChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(null);

  const API_BASE = "https://alaska-rag-api-697768193921.us-central1.run.app";

  const samples = [
    "What are snow removal procedures?",
    "How do I report road hazards?",
    "When do emergency shelters open?",
    "What are winter emergency protocols?",
  ];

  // Check API status on mount
  useEffect(() => {
    fetch(API_BASE + "/")
      .then((res) => setApiOnline(res.ok))
      .catch(() => setApiOnline(false));
  }, []);

  const addMessage = (type, content, error = false) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        type,
        content,
        error,
        time: new Date().toLocaleTimeString(),
      },
    ]);
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!input.trim() || isLoading) return;

    const question = input.trim();
    setInput("");
    addMessage("user", question);
    setIsLoading(true);

    try {
      const response = await fetch(API_BASE + "/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();

      if (response.ok) {
        addMessage("bot", data.answer);
      } else {
        throw new Error(data.detail || "Request failed");
      }
    } catch (error) {
      addMessage(
        "bot",
        "Sorry, I encountered an error. Please try again later.",
        true
      );
    } finally {
      setIsLoading(false);
    }
  };

  const StatusDot = () => (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "8px",
        fontSize: "14px",
        color: "#666",
      }}
    >
      <div
        style={{
          width: "8px",
          height: "8px",
          borderRadius: "50%",
          backgroundColor:
            apiOnline === null ? "#ffa500" : apiOnline ? "#4caf50" : "#f44336",
        }}
      />
      API{" "}
      {apiOnline === null ? "Checking..." : apiOnline ? "Online" : "Offline"}
    </div>
  );

  const Message = ({ msg }) => (
    <div
      style={{
        display: "flex",
        gap: "12px",
        marginBottom: "16px",
        alignItems: "flex-start",
      }}
    >
      <div
        style={{
          width: "32px",
          height: "32px",
          borderRadius: "50%",
          backgroundColor: msg.type === "user" ? "#2196f3" : "#4caf50",
          color: "white",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "14px",
          fontWeight: "bold",
          flexShrink: 0,
        }}
      >
        {msg.type === "user" ? "U" : "AI"}
      </div>
      <div
        style={{
          backgroundColor: msg.error ? "#ffebee" : "#f5f5f5",
          padding: "12px 16px",
          borderRadius: "12px",
          maxWidth: "80%",
          color: msg.error ? "#c62828" : "#333",
        }}
      >
        {msg.content}
      </div>
    </div>
  );

  const LoadingMessage = () => (
    <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
      <div
        style={{
          width: "32px",
          height: "32px",
          borderRadius: "50%",
          backgroundColor: "#4caf50",
          color: "white",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "14px",
          fontWeight: "bold",
        }}
      >
        AI
      </div>
      <div
        style={{
          backgroundColor: "#f5f5f5",
          padding: "12px 16px",
          borderRadius: "12px",
          display: "flex",
          gap: "4px",
        }}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: "#666",
              animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div
      style={{
        maxWidth: "800px",
        margin: "0 auto",
        padding: "20px",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        minHeight: "100vh",
        backgroundColor: "#fafafa",
      }}
    >
      <style>{`
        @keyframes pulse {
          0%, 80%, 100% { opacity: 0.3; }
          40% { opacity: 1; }
        }
        
        .sample-btn {
          background: white;
          border: 1px solid #ddd;
          padding: 8px 12px;
          border-radius: 20px;
          cursor: pointer;
          font-size: 13px;
          transition: all 0.2s;
        }
        
        .sample-btn:hover {
          background: #f0f0f0;
          border-color: #2196f3;
        }
      `}</style>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "30px" }}>
        <h1
          style={{
            color: "#1976d2",
            marginBottom: "8px",
            fontSize: "28px",
          }}
        >
          ❄️ Alaska Emergency Services
        </h1>
        <p style={{ color: "#666", marginBottom: "16px" }}>
          Get answers about snow removal, road conditions, and emergency
          protocols
        </p>
        <StatusDot />
      </div>

      {/* Chat Area */}
      <div
        style={{
          backgroundColor: "white",
          borderRadius: "12px",
          padding: "24px",
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
          marginBottom: "20px",
          minHeight: "400px",
        }}
      >
        {messages.length === 0 ? (
          <div style={{ textAlign: "center", color: "#666" }}>
            <h3 style={{ marginBottom: "16px" }}>
              Welcome! How can I help you today?
            </h3>
            <p style={{ marginBottom: "20px" }}>
              Try asking about winter services or emergency procedures:
            </p>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "8px",
                justifyContent: "center",
              }}
            >
              {samples.map((sample, i) => (
                <button
                  key={i}
                  className="sample-btn"
                  onClick={() => setInput(sample)}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div style={{ maxHeight: "500px", overflowY: "auto" }}>
            {messages.map((msg) => (
              <Message key={msg.id} msg={msg} />
            ))}
            {isLoading && <LoadingMessage />}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div style={{ display: "flex", gap: "12px" }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && handleSubmit(e)}
          placeholder="Ask about snow removal, emergencies, or road conditions..."
          disabled={isLoading || apiOnline === false}
          style={{
            flex: 1,
            padding: "12px 16px",
            border: "1px solid #ddd",
            borderRadius: "24px",
            fontSize: "14px",
            outline: "none",
            backgroundColor:
              isLoading || apiOnline === false ? "#f5f5f5" : "white",
          }}
        />
        <button
          onClick={handleSubmit}
          disabled={isLoading || !input.trim() || apiOnline === false}
          style={{
            padding: "12px 24px",
            backgroundColor:
              isLoading || !input.trim() || apiOnline === false
                ? "#ccc"
                : "#2196f3",
            color: "white",
            border: "none",
            borderRadius: "24px",
            cursor:
              isLoading || !input.trim() || apiOnline === false
                ? "not-allowed"
                : "pointer",
            fontSize: "14px",
            fontWeight: "600",
          }}
        >
          {isLoading ? "..." : "Send →"}
        </button>
      </div>

      {/* Quick suggestions when no messages */}
      {messages.length === 0 && (
        <div
          style={{
            marginTop: "16px",
            textAlign: "center",
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <span style={{ color: "#666", fontSize: "13px" }}>Quick start:</span>
          {samples.slice(0, 2).map((sample, i) => (
            <button
              key={i}
              className="sample-btn"
              onClick={() => setInput(sample)}
            >
              {sample}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default AlaskaChat;
