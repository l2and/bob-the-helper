# Bob Ross Helper - Friction Log
*A personal perspective on building with LangChain & LangGraph*

**Project:** LangChain Documentation Assistant Chrome Extension  
---

## 🎯 **What Is This Document?**

This friction log captures every frustrating moment, unclear documentation, and "why doesn't this work?!" experience I had while building a real LangChain application. These are some of the painpoints I ran into along with some tips on how to get through it. As a Langchain beginner, there was a lot of catch up and learning needed.

---


## **Setting up Evaluation SDK**
**Problem:** There are many solutions to setting up different types of evaluations along with their examples.
**Specific Issue:**
- In [evaluating a chatbot](https://docs.langchain.com/langsmith/evaluate-chatbot-tutorial), you can see the process of how to build the dataset to finally running the evaluation. For each example, you just have to follow each of the instructions, and it's a bit of a hassle for those who learn better by seeing how something works.
**Customer Impact:** LOW - Walking through each example just slows down ramp up time.  

**Recommendation:** Create workable notebooks so that beginners can see what's going on in a working example for each tutorial.

---

### **General Anthropic API documentation lacking**
**Problem:** Langchain docs primarily focus on utilizing OpenAI  

**Solution:**
When reading into example code, you must doublecheck the [Anthropic API docs](https://python.langchain.com/api_reference/anthropic/index.html) just to see that the same feature is there as in OpenAI

---

## 🔧 **Development Environment Friction**

### **General Local environment setup**
**Problem:** I first looked for a specific Docker container that would make life a bit easier to spin up a langgraph instance
**Pain Point:** The Documentation is pretty convoluted and I settled on just installing and running everything locally 

---

## 🎯 **What Worked Really Well**

### **Smooth Experiences:**
- **LangChain Expression Language (LCEL)** - Intuitive pipe syntax
- **Anthropic integration** - Clean API, good error messages
- **LangSmith tracing** - Automatic and helpful for debugging
- **LangGraph visualization** - Great for understanding workflow

### **Excellent Documentation:**
- Basic LangChain tutorials and quickstarts
- LangGraph core concepts (nodes, edges, state)
