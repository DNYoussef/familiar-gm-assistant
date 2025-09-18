"""Financial BERT Model

Specialized BERT model for financial text understanding.
Fine-tuned on financial news, earnings calls, and market commentary.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple
import numpy as np


class FinancialBERT(nn.Module):
    """BERT model specialized for financial text analysis.

    Fine-tuned on financial corpora with domain-specific vocabulary
    and market-aware attention patterns.
    """

    def __init__(self,
                 model_name: str = 'bert-base-uncased',
                 num_classes: int = 3,  # Negative, Neutral, Positive
                 dropout_rate: float = 0.1,
                 max_length: int = 512):
        """Initialize FinancialBERT.

        Args:
            model_name: Pre-trained BERT model name
            num_classes: Number of sentiment classes
            dropout_rate: Dropout probability
            max_length: Maximum sequence length
        """
        super(FinancialBERT, self).__init__()

        self.max_length = max_length
        self.num_classes = num_classes

        # Load pre-trained BERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        # Financial domain adaptation layers
        self.financial_adapter = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_rate)
        )

        # Multi-task prediction heads
        self.sentiment_head = nn.Linear(self.config.hidden_size, num_classes)

        self.volatility_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size // 2, 1),
            nn.Sigmoid()  # Volatility impact [0, 1]
        )

        self.urgency_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size // 4, 1),
            nn.Sigmoid()  # Urgency score [0, 1]
        )

        # Financial entity attention
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )

        # Market impact predictor
        self.market_impact_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size // 2, 5),  # [Very Negative, Negative, Neutral, Positive, Very Positive]
            nn.Softmax(dim=-1)
        )

        # Initialize new layers
        self._init_custom_weights()

        # Financial vocabulary extensions
        self.financial_terms = self._load_financial_vocabulary()

    def _init_custom_weights(self):
        """Initialize custom layer weights."""
        for module in [self.financial_adapter, self.sentiment_head,
                      self.volatility_head, self.urgency_head, self.market_impact_head]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _load_financial_vocabulary(self) -> Dict[str, float]:
        """Load financial domain vocabulary with sentiment weights."""
        return {
            # Positive financial terms
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'gains': 0.6,
            'breakout': 0.7, 'uptrend': 0.6, 'momentum': 0.5, 'outperform': 0.7,
            'beat expectations': 0.8, 'strong earnings': 0.8, 'upgrade': 0.7,
            'buy rating': 0.8, 'accumulate': 0.6, 'positive outlook': 0.6,

            # Negative financial terms
            'bearish': -0.8, 'crash': -0.9, 'plunge': -0.8, 'losses': -0.6,
            'breakdown': -0.7, 'downtrend': -0.6, 'sell-off': -0.7, 'underperform': -0.7,
            'miss expectations': -0.8, 'weak earnings': -0.8, 'downgrade': -0.7,
            'sell rating': -0.8, 'distribute': -0.6, 'negative outlook': -0.6,

            # Volatility terms
            'volatile': 0.0, 'uncertainty': -0.3, 'risk': -0.2, 'turbulence': -0.4,
            'correction': -0.3, 'pullback': -0.2, 'consolidation': 0.1,

            # Market structure terms
            'support': 0.2, 'resistance': -0.1, 'volume': 0.0, 'liquidity': 0.1,
            'institutional': 0.2, 'retail': -0.1, 'algorithmic': 0.0
        }

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through FinancialBERT.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]

        Returns:
            Dictionary containing all prediction outputs
        """
        # BERT forward pass
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Get sequence output and pooled output
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = bert_outputs.pooler_output        # [batch_size, hidden_size]

        # Financial domain adaptation
        adapted_output = self.financial_adapter(pooled_output)

        # Apply entity attention to sequence output
        attended_output, attention_weights = self.entity_attention(
            sequence_output, sequence_output, sequence_output,
            key_padding_mask=~attention_mask.bool()
        )

        # Global context from attention
        global_context = attended_output.mean(dim=1)  # [batch_size, hidden_size]

        # Combine adapted pooled output with global context
        final_representation = (adapted_output + global_context) / 2

        # Generate predictions
        sentiment_logits = self.sentiment_head(final_representation)
        volatility_impact = self.volatility_head(final_representation)
        urgency_score = self.urgency_head(final_representation)
        market_impact = self.market_impact_head(final_representation)

        return {
            'sentiment_logits': sentiment_logits,
            'sentiment_probs': torch.softmax(sentiment_logits, dim=-1),
            'volatility_impact': volatility_impact,
            'urgency_score': urgency_score,
            'market_impact_probs': market_impact,
            'attention_weights': attention_weights,
            'final_representation': final_representation,
            'sequence_output': sequence_output
        }

    def predict_sentiment(self, text: str, tokenizer) -> Dict[str, float]:
        """Predict sentiment for a single text.

        Args:
            text: Input text
            tokenizer: BERT tokenizer

        Returns:
            Sentiment predictions and scores
        """
        self.eval()

        with torch.no_grad():
            # Tokenize input
            inputs = tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            # Forward pass
            outputs = self.forward(
                inputs['input_ids'],
                inputs['attention_mask'],
                inputs.get('token_type_ids')
            )

            # Extract predictions
            sentiment_probs = outputs['sentiment_probs'][0].cpu().numpy()
            volatility = float(outputs['volatility_impact'][0].cpu().numpy())
            urgency = float(outputs['urgency_score'][0].cpu().numpy())
            market_impact = outputs['market_impact_probs'][0].cpu().numpy()

            # Calculate composite sentiment score
            sentiment_labels = ['negative', 'neutral', 'positive']
            sentiment_scores = {label: float(prob) for label, prob in zip(sentiment_labels, sentiment_probs)}

            # Overall sentiment (-1 to 1)
            overall_sentiment = sentiment_probs[2] - sentiment_probs[0]  # positive - negative

            return {
                'sentiment_scores': sentiment_scores,
                'overall_sentiment': float(overall_sentiment),
                'predicted_class': sentiment_labels[sentiment_probs.argmax()],
                'confidence': float(sentiment_probs.max()),
                'volatility_impact': volatility,
                'urgency_score': urgency,
                'market_impact': {
                    'very_negative': float(market_impact[0]),
                    'negative': float(market_impact[1]),
                    'neutral': float(market_impact[2]),
                    'positive': float(market_impact[3]),
                    'very_positive': float(market_impact[4])
                },
                'financial_terms_detected': self._detect_financial_terms(text)
            }

    def _detect_financial_terms(self, text: str) -> List[Dict[str, float]]:
        """Detect financial terms in text with sentiment weights.

        Args:
            text: Input text

        Returns:
            List of detected terms with weights
        """
        text_lower = text.lower()
        detected_terms = []

        for term, weight in self.financial_terms.items():
            if term in text_lower:
                detected_terms.append({
                    'term': term,
                    'weight': weight,
                    'positions': [i for i in range(len(text_lower)) if text_lower.startswith(term, i)]
                })

        return detected_terms

    def extract_financial_entities(self,
                                 text: str,
                                 tokenizer,
                                 entity_threshold: float = 0.5) -> List[Dict]:
        """Extract financial entities using attention patterns.

        Args:
            text: Input text
            tokenizer: BERT tokenizer
            entity_threshold: Attention threshold for entity extraction

        Returns:
            List of extracted entities with attention scores
        """
        self.eval()

        with torch.no_grad():
            inputs = tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors='pt',
                return_offsets_mapping=True
            )

            outputs = self.forward(
                inputs['input_ids'],
                inputs['attention_mask']
            )

            # Get attention weights and tokens
            attention_weights = outputs['attention_weights'][0]  # [seq_len, seq_len]
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            # Average attention across heads and positions
            token_importance = attention_weights.mean(dim=0).mean(dim=0).cpu().numpy()

            # Extract high-attention tokens as entities
            entities = []
            for i, (token, importance) in enumerate(zip(tokens, token_importance)):
                if importance > entity_threshold and not token.startswith('['):
                    entities.append({
                        'token': token,
                        'position': i,
                        'attention_score': float(importance),
                        'is_financial_term': token.lower() in self.financial_terms
                    })

            return entities

    def fine_tune_on_financial_data(self,
                                   train_dataloader,
                                   val_dataloader,
                                   num_epochs: int = 3,
                                   learning_rate: float = 2e-5):
        """Fine-tune model on financial text data.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.7
        )

        # Multi-task loss function
        sentiment_criterion = nn.CrossEntropyLoss()
        regression_criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0

            for batch in train_dataloader:
                optimizer.zero_grad()

                outputs = self.forward(
                    batch['input_ids'],
                    batch['attention_mask'],
                    batch.get('token_type_ids')
                )

                # Multi-task loss
                sentiment_loss = sentiment_criterion(
                    outputs['sentiment_logits'],
                    batch['sentiment_labels']
                )

                volatility_loss = regression_criterion(
                    outputs['volatility_impact'].squeeze(),
                    batch.get('volatility_targets', torch.zeros_like(outputs['volatility_impact'].squeeze()))
                )

                total_loss = sentiment_loss + 0.3 * volatility_loss
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += total_loss.item()

            # Validation phase
            self.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_dataloader:
                    outputs = self.forward(
                        batch['input_ids'],
                        batch['attention_mask'],
                        batch.get('token_type_ids')
                    )

                    sentiment_loss = sentiment_criterion(
                        outputs['sentiment_logits'],
                        batch['sentiment_labels']
                    )
                    val_loss += sentiment_loss.item()

            scheduler.step(val_loss)

            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss/len(train_dataloader):.4f}')
            print(f'  Val Loss: {val_loss/len(val_dataloader):.4f}')


def create_financial_bert(model_name: str = 'bert-base-uncased') -> FinancialBERT:
    """Create FinancialBERT model with pre-trained weights.

    Args:
        model_name: Pre-trained BERT model name

    Returns:
        Initialized FinancialBERT model
    """
    model = FinancialBERT(model_name=model_name)

    # Load financial domain vocabulary
    print(f\"Loaded {len(model.financial_terms)} financial terms\")

    return model"