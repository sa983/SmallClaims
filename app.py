from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import json
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the knowledge base (structured from the provided document)
class KnowledgeBase:
    def __init__(self):
        self.sections = {
            "general_procedure": {
                "title": "General Procedure",
                "content": """
                Small Claims Court is a specialized court designed to provide an accessible and simplified process for resolving 
                minor disputes. Key features include: monetary limit of $5,000 for Ithaca City Court; informal procedures; 
                optional legal representation; handles various case types including unpaid debts, property damage, and breach of 
                contract; lower filing fees ($15-$20); hearings before a judge; no jury trials unless requested by defendant.
                """,
                "subsections": {
                    "filing": {
                        "title": "Filing a Claim",
                        "content": """
                        To file a claim, complete an application form with accurate names and addresses of all parties and a 
                        description of events. Filing fees in Ithaca City Court are $15 for claims up to $1,000 and $20 for claims 
                        between $1,000-$5,000. Partnerships can only initiate commercial small claims. Corporations, LLCs, and 
                        associations must file commercial small claims.
                        """
                    },
                    "serving": {
                        "title": "Serving the Defendant",
                        "content": """
                        You must serve the defendant with a copy of the summons and complaint. This can be done through personal 
                        service (someone over 18 not involved in the case delivers documents), certified mail with return receipt, 
                        or substitute service (leaving documents with someone at defendant's home/business). The person serving 
                        must complete an affidavit of service.
                        """
                    },
                    "trial": {
                        "title": "Day of Trial",
                        "content": """
                        Arrive early, bring all relevant documents and evidence, organize materials logically, dress professionally, 
                        prepare a clear opening statement, ensure witnesses are present, and maintain a respectful demeanor. 
                        During the trial, you present your case first, followed by the defendant. Both sides can question each 
                        other and any witnesses. The court typically mails its decision within 30 days.
                        """
                    }
                }
            },
            "collections": {
                "title": "Collections",
                "content": """
                If you win a claim, you become the judgment creditor. Judgments in NY are valid for 20 years with 9% annual interest. 
                First contact the judgment debtor to ensure they're aware of the court's decision. If not paid within 30 days, 
                you may begin collection efforts such as garnishing wages, seizing assets, placing liens on property, or suspending licenses.
                """,
                "subsections": {
                    "information_subpoena": {
                        "title": "Information Subpoena",
                        "content": """
                        An information subpoena identifies the location of debtor's assets. It can be sent to the debtor or any 
                        entity with information about their assets. You can obtain one from your local court clerk for $3. 
                        It must be served by registered/certified mail with return receipt.
                        """
                    },
                    "enforcement_officers": {
                        "title": "Enforcement Officers",
                        "content": """
                        For uncooperative debtors, you may need a sheriff or city marshal to help collect the debt. 
                        Contact an officer in a county where the debtor has property. Provide information about the debtor's 
                        assets and ask them to obtain an "execution" to seize property or money. Their fees may be added to 
                        the judgment amount.
                        """
                    }
                }
            },
            "landlord_tenant": {
                "title": "Landlord/Tenant Law",
                "content": """
                In New York, landlord-tenant relationships are governed by Article 7 of the Consolidated Laws. Leases must identify 
                premises, parties, rent amount, duration, and rights/obligations. Certain lease provisions are illegal, including 
                exempting landlords from liability or waiving habitability warranty. Rent regulation includes rent control and rent stabilization.
                """,
                "subsections": {
                    "security_deposits": {
                        "title": "Security Deposits",
                        "content": """
                        Landlords can require up to one month's rent as security deposit. Buildings with 6+ units must place deposits 
                        in interest-bearing accounts. Deposits must be returned with itemized deductions within 14 days of move-out.
                        If not provided on time, landlord must return entire deposit regardless of damage.
                        """
                    },
                    "evictions": {
                        "title": "Evictions",
                        "content": """
                        Landlords must give 14-day written notice for non-payment before eviction proceedings. Only a sheriff, 
                        marshal, or constable can execute court-ordered eviction warrants. Tenants can dismiss non-payment cases 
                        by paying all owed rent until actual eviction. Tenants cannot be evicted for non-payment of fees like late fees.
                        """
                    }
                }
            },
            "auto_law": {
                "title": "Auto Law",
                "content": """
                New York's lemon laws protect consumers who purchase/lease cars that don't meet standards. New cars are those purchased/leased 
                less than 2 years from original delivery with fewer than 18,000 miles. Used car lemon law applies to dealer sales with purchase 
                price of at least $1,500 and up to 100,000 miles.
                """,
                "subsections": {
                    "car_accidents": {
                        "title": "Car Accidents",
                        "content": """
                        New York is a no-fault insurance state with $50,000 minimum coverage for medical costs and limited lost income. 
                        You can sue for economic damages beyond no-fault benefits and for non-economic damages only for "serious injury" 
                        as defined by Insurance Law Section 5102. Cases are based on negligence (duty, breach, causation, damages).
                        """
                    },
                    "repairs": {
                        "title": "Auto Repairs",
                        "content": """
                        Deal only with registered shops (green and white "Registered State of New York Motor Vehicle Repair Shop" sign). 
                        Request written estimates listing parts, costs, and labor charges. Shops cannot perform work without permission or 
                        charge more than estimated without approval. You're entitled to all replaced parts if requested in writing before work.
                        """
                    }
                }
            },
            "statute_limitations": {
                "title": "Statute of Limitations",
                "content": """
                Different types of cases have different time limits for filing. Key time limits include: contracts (written or oral) - 6 years; 
                property damage - 3 years; car accidents - 3 years; medical malpractice - 2 years and 6 months; debt collection - 3 years; 
                fraud - 6 years.
                """
            }
        }
        
        # Define citations for different sections
        self.citations = {
            "general_procedure": "New York State Unified Court System, Small Claims Court Guide",
            "collections": "New York State Unified Court System, Collecting Judgments",
            "landlord_tenant": "New York State Real Property Law",
            "auto_law": "New York State Attorney General's Office, Consumer Guides",
            "statute_limitations": "New York CPLR (Civil Practice Law and Rules)"
        }
    
    def preprocess_text(self, text):
        # Tokenize, remove punctuation, lemmatize, remove stop words
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token not in stop_words]
        return tokens
    
    def search(self, query):
        query_tokens = self.preprocess_text(query)
        results = []
        
        # Search in main sections
        for section_key, section in self.sections.items():
            section_tokens = self.preprocess_text(section["content"])
            score = self.calculate_relevance(query_tokens, section_tokens)
            if score > 0.2:  # Threshold for relevance
                results.append({
                    "section": section["title"],
                    "content": section["content"].strip(),
                    "score": score,
                    "citation": self.citations.get(section_key, "New York State Law")
                })
            
            # Search in subsections if they exist
            if "subsections" in section:
                for subsec_key, subsec in section["subsections"].items():
                    subsec_tokens = self.preprocess_text(subsec["content"])
                    score = self.calculate_relevance(query_tokens, subsec_tokens)
                    if score > 0.3:  # Higher threshold for subsections
                        results.append({
                            "section": f"{section['title']} - {subsec['title']}",
                            "content": subsec["content"].strip(),
                            "score": score,
                            "citation": self.citations.get(section_key, "New York State Law")
                        })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:3]  # Return top 3 most relevant results
    
    def calculate_relevance(self, query_tokens, section_tokens):
        # Simple relevance calculation based on token overlap
        query_set = set(query_tokens)
        section_set = set(section_tokens)
        
        if not query_set or not section_set:
            return 0
        
        intersection = query_set.intersection(section_set)
        return len(intersection) / len(query_set)
    
    def get_section_titles(self):
        titles = []
        for section_key, section in self.sections.items():
            titles.append(section["title"])
            if "subsections" in section:
                for subsec_key, subsec in section["subsections"].items():
                    titles.append(f"{section['title']} - {subsec['title']}")
        return titles

# Initialize knowledge base
kb = KnowledgeBase()

@app.route('/')
def home():
    # Get all section titles for the navigation menu
    sections = kb.get_section_titles()
    return render_template('index.html', sections=sections)

@app.route('/api/ask', methods=['POST'])
def ask():
    user_question = request.json.get('question', '')
    
    if not user_question:
        return jsonify({"error": "No question provided"}), 400
    
    results = kb.search(user_question)
    
    if not results:
        return jsonify({
            "answer": "I don't have enough information to answer your question. Please try asking something about small claims court procedures, collections, landlord-tenant law, auto law, or statute of limitations in New York State.",
            "results": []
        })
    
    # Construct a helpful response
    answer = f"Based on your question about {user_question}, here's what I found:\n\n"
    for result in results:
        answer += f"{result['content']}\n\nSource: {result['citation']}\n\n"
    
    return jsonify({
        "answer": answer,
        "results": results
    })

# Create templates directory and files
os.makedirs('templates', exist_ok=True)

# Create index.html
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NY Small Claims Court Helper</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            padding-top: 20px;
        }
        .chat-container {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .user-message {
            background-color: #e6f7ff;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            max-width: 75%;
            margin-left: auto;
        }
        .bot-message {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            max-width: 75%;
        }
        .citation {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        .sidebar {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4">
            <div class="col">
                <h1 class="text-center">NY Small Claims Court Helper</h1>
                <p class="text-center">Ask questions about small claims court procedures in New York State</p>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-8">
                <div class="chat-container" id="chatContainer">
                    <div class="bot-message">
                        Welcome to the NY Small Claims Court Helper! How can I assist you today? You can ask me about:
                        <ul>
                            <li>General small claims court procedures</li>
                            <li>Filing a claim</li>
                            <li>Serving defendants</li>
                            <li>The trial process</li>
                            <li>Collecting judgments</li>
                            <li>And more...</li>
                        </ul>
                    </div>
                </div>
                <div class="input-group mb-3">
                    <input type="text" id="userQuestion" class="form-control" placeholder="Ask a question...">
                    <button class="btn btn-primary" type="button" id="sendButton">Send</button>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="sidebar">
                    <h5>Topics</h5>
                    <ul class="list-group">
                        {% for section in sections %}
                        <li class="list-group-item" style="cursor: pointer;" onclick="suggestQuestion('{{ section }}')">{{ section }}</li>
                        {% endfor %}
                    </ul>
                    
                    <h5 class="mt-4">Common Questions</h5>
                    <ul class="list-group">
                        <li class="list-group-item" style="cursor: pointer;" onclick="suggestQuestion('How do I file a small claims case?')">How do I file a small claims case?</li>
                        <li class="list-group-item" style="cursor: pointer;" onclick="suggestQuestion('What is the monetary limit for small claims court?')">What is the monetary limit for small claims court?</li>
                        <li class="list-group-item" style="cursor: pointer;" onclick="suggestQuestion('How do I collect a judgment?')">How do I collect a judgment?</li>
                        <li class="list-group-item" style="cursor: pointer;" onclick="suggestQuestion('What should I do on the day of trial?')">What should I do on the day of trial?</li>
                        <li class="list-group-item" style="cursor: pointer;" onclick="suggestQuestion('How long do I have to file my case?')">How long do I have to file my case?</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('userQuestion').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
        
        document.getElementById('sendButton').addEventListener('click', askQuestion);
        
        function askQuestion() {
            const userInput = document.getElementById('userQuestion').value.trim();
            if (!userInput) return;
            
            // Add user message to chat
            addMessage(userInput, 'user');
            
            // Clear input field
            document.getElementById('userQuestion').value = '';
            
            // Show loading message
            const loadingId = addMessage('Searching for information...', 'bot');
            
            // Send question to server
            fetch('/api/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: userInput }),
            })
            .then(response => response.json())
            .then(data => {
                // Remove loading message
                document.getElementById(loadingId).remove();
                
                // Add bot response
                addMessage(data.answer, 'bot');
            })
            .catch(error => {
                // Remove loading message
                document.getElementById(loadingId).remove();
                
                // Add error message
                addMessage('Sorry, there was an error processing your question. Please try again.', 'bot');
                console.error('Error:', error);
            });
        }
        
        function addMessage(text, sender) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            const messageId = 'msg-' + Date.now();
            messageDiv.id = messageId;
            
            if (sender === 'user') {
                messageDiv.className = 'user-message';
                messageDiv.innerText = text;
            } else {
                messageDiv.className = 'bot-message';
                messageDiv.innerHTML = text.replace(/\\n/g, '<br>');
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            return messageId;
        }
        
        function suggestQuestion(text) {
            document.getElementById('userQuestion').value = text;
            document.getElementById('userQuestion').focus();
        }
    </script>
</body>
</html>
    ''')

# For running the app locally
if __name__ == '__main__':
    app.run(debug=True)
