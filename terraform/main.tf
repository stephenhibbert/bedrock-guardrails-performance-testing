terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

provider "aws" {
  region = var.aws_region
}

resource "aws_bedrock_guardrail" "example" {
  name                      = "my-test-guardrail"
  description               = "My Test Guardrail"
  blocked_input_messaging   = "This input has been blocked due to security policies."
  blocked_outputs_messaging = "This output has been blocked due to security policies."

  # Content Policy - Enable all filters at maximum strength
  content_policy_config {
    dynamic "filters_config" {
      for_each = [ "PROMPT_ATTACK", "SEXUAL", "VIOLENCE", "HATE", "INSULTS", "MISCONDUCT" ]
      content {
        type           = filters_config.value
        input_strength  = "HIGH"
        output_strength = "NONE"
      }
    }
  }

  # Sensitive Information Policy - Block all PII types
  # sensitive_information_policy_config {
  #   dynamic "pii_entities_config" {
  #     for_each = [
  #       "ADDRESS", "AGE", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "CA_HEALTH_NUMBER",
  #       "CA_SOCIAL_INSURANCE_NUMBER", "CREDIT_DEBIT_CARD_CVV", "CREDIT_DEBIT_CARD_EXPIRY",
  #       "CREDIT_DEBIT_CARD_NUMBER", "DRIVER_ID", "EMAIL", "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
  #       "IP_ADDRESS", "LICENSE_PLATE", "MAC_ADDRESS", "NAME", "PASSWORD", "PHONE",
  #       "PIN", "SWIFT_CODE", "UK_NATIONAL_HEALTH_SERVICE_NUMBER", "UK_NATIONAL_INSURANCE_NUMBER",
  #       "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER", "URL", "USERNAME", "US_BANK_ACCOUNT_NUMBER",
  #       "US_BANK_ROUTING_NUMBER", "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
  #       "US_PASSPORT_NUMBER", "US_SOCIAL_SECURITY_NUMBER", "VEHICLE_IDENTIFICATION_NUMBER"
  #     ]
  #     content {
  #       type   = pii_entities_config.value
  #       action = "BLOCK"
  #     }
  #   }

    # Add some common regex patterns
  #   regexes_config {
  #     name        = "credit_card"
  #     description = "Credit card number pattern"
  #     pattern     = "^4[0-9]{12}(?:[0-9]{3})?$"
  #     action      = "BLOCK"
  #   }
  # }

  # Topic Policy - Block sensitive topics
  # topic_policy_config {
  #   topics_config {
  #     name       = "financial_advice"
  #     definition = "Any content related to financial advice or investment recommendations"
  #     examples   = ["Where should I invest my money?", "What stocks should I buy?"]
  #     type       = "DENY"
  #   }
  #   topics_config {
  #     name       = "medical_advice"
  #     definition = "Any content providing medical advice or diagnoses"
  #     examples   = ["What medication should I take?", "How do I treat this condition?"]
  #     type       = "DENY"
  #   }
  #   topics_config {
  #     name       = "legal_advice"
  #     definition = "Any content providing legal advice or recommendations"
  #     examples   = ["How should I structure my will?", "What are my legal rights?"]
  #     type       = "DENY"
  #   }
  # }

  # Word Policy - Block profanity and custom words
  # word_policy_config {
  #   managed_word_lists_config {
  #     type = "PROFANITY"
  #   }
  #   words_config {
  #     text = "CONFIDENTIAL"
  #   }
  #   words_config {
  #     text = "SECRET"
  #   }
  #   words_config {
  #     text = "CLASSIFIED"
  #   }
  # }

  contextual_grounding_policy_config {
    filters_config {
      type      = "GROUNDING"
      threshold = 0.8
    }
    # filters_config {
    #   type      = "RELEVANCE"
    #   threshold = 0.7
    # }
  }

  tags = {
    Environment = "dev"
    Managed_By  = "terraform"
  }
}

output "guardrail_id" {
  description = "ID of the created guardrail"
  value       = aws_bedrock_guardrail.example.guardrail_id
}

output "guardrail_arn" {
  description = "ARN of the created guardrail"
  value       = aws_bedrock_guardrail.example.guardrail_arn
}

output "guardrail_version" {
  description = "Version of the created guardrail"
  value       = aws_bedrock_guardrail.example.version
}

output "aws_region" {
  description = "AWS region"
  value       = var.aws_region
}

resource "aws_bedrock_guardrail_version" "version-example" {
  description   = "1"
  guardrail_arn = aws_bedrock_guardrail.example.guardrail_arn
  skip_destroy  = true
}