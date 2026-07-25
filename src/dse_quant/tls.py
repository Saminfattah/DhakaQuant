from __future__ import annotations

import logging

import truststore

LOGGER = logging.getLogger(__name__)


def configure_tls(use_system_ca_store: bool) -> None:
    """Use the operating-system certificate store while retaining TLS verification."""
    if use_system_ca_store:
        truststore.inject_into_ssl()
        LOGGER.debug("HTTPS validation is using the operating-system certificate store.")

