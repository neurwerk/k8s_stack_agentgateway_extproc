"""Protect native-format routing and retained upload limits."""

import base64
import io
import zipfile
from email import policy
from email.parser import BytesParser

import httpx
import pytest
from pydantic import SecretStr

from agentgateway_extproc.config.settings import DoclingSettings
from agentgateway_extproc.lib.docling import DoclingClient
from agentgateway_extproc.lib.documents import _MIMES, DocumentError, preflight
from agentgateway_extproc.models.engine import EngineAttachmentPart

from .test_documents import _upload


def part(extension, data):
    return EngineAttachmentPart.model_validate(
        {
            "type": "file",
            "file": {
                "filename": f"sample.{extension}",
                "file_data": f"data:{_MIMES[extension]};base64," + base64.b64encode(data).decode(),
            },
        }
    )


@pytest.mark.parametrize(
    "extension,data,backend",
    [
        ("html", b"<html><body><p>Example</p></body></html>", "html"),
        ("htm", b"<html><body><p>Example</p></body></html>", "html"),
        ("asciidoc", b"= Example\n\nBody", "asciidoc"),
        ("adoc", b"= Example\n\nBody", "asciidoc"),
        ("ics", b"BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n", None),
        ("eml", b"From: test@example.org\r\nSubject: Example\r\n\r\nBody", None),
        ("mhtml", b"MIME-Version: 1.0\r\nContent-Type: text/html\r\n\r\n<p>Body</p>", None),
        ("rtf", rb"{\rtf1\ansi Example}", None),
        ("tex", rb"\documentclass{article}\begin{document}Example\end{document}", None),
        ("vtt", b"WEBVTT\n\n00:00.000 --> 00:01.000\nExample\n", "vtt"),
        ("boxnote", b'{"atext":{"text":"Example","attribs":""},"pool":{}}', None),
        ("nxml", b"<article><body><p>Example</p></body></article>", None),
    ],
)
@pytest.mark.asyncio
async def test_native_extension_mime_and_backend_are_preserved(extension, data, backend):
    upload = preflight([part(extension, data)], DoclingSettings())[0]
    requests = []

    def respond(request):
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + request.headers["content-type"].encode()
            + b"\r\n\r\n"
            + request.content
        )
        fields = {
            p.get_param("name", header="content-disposition"): p for p in message.iter_parts()
        }
        assert fields["files"].get_filename() == f"upload.{extension}"
        assert fields["files"]["Content-Type"] == _MIMES[extension]
        boundary_text = message.get_boundary()
        assert boundary_text is not None
        boundary = boundary_text.encode()
        file_part = next(
            p
            for p in request.content.split(b"--" + boundary)
            if b'name="files"' in p.partition(b"\r\n\r\n")[0]
        )
        assert file_part.partition(b"\r\n\r\n")[2] == data + b"\r\n"
        if backend is None:
            assert "from_formats" not in fields
        else:
            assert fields["from_formats"].get_payload(decode=True) == backend.encode()
        requests.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        docling = DoclingClient(
            DoclingSettings(base_url="https://docling.test", api_key=SecretStr("test")), client
        )
        await docling._http("POST", "/v1/convert/file/async", limit=1024, upload=upload)
    assert len(requests) == 1


@pytest.mark.parametrize("data", [b"", b" \n", b"\xff", b"a\x00b", b"%PDF-fake", b"PK\x03\x04fake"])
def test_new_text_formats_retain_decode_checks(data):
    with pytest.raises((DocumentError, UnicodeDecodeError)):
        preflight([part("txt", b"valid first"), part("html", data)], DoclingSettings())


@pytest.mark.parametrize("extension", ["xlsx", "xlsm"])
@pytest.mark.parametrize(
    "case,status",
    [
        ("ok", None),
        ("zip-path", 400),
        ("zip-part", 400),
        ("zip-limit", 413),
        ("xlsx-sparse", 413),
        ("xlsx-dimension", 413),
        ("xlsx-merge", 413),
        ("xlsx-doctype", 400),
        ("xlsx-chart", 413),
        ("xlsx-hyperlink", 413),
        ("xlsx-repeated-hyperlinks", 413),
    ],
)
def test_spreadsheet_alias_retains_package_and_expansion_limits(extension, case, status):
    source = _upload("xlsx", case)
    data = base64.b64decode(source["file_data"].partition(",")[2])
    if status is None:
        assert preflight([part(extension, data)], DoclingSettings())[0].format == extension
        assert DoclingClient(DoclingSettings())._options(extension)["from_formats"] == "xlsx"
    else:
        with pytest.raises(DocumentError) as error:
            preflight([part(extension, data)], DoclingSettings())
        assert error.value.status == status


@pytest.mark.parametrize("root", ["word/document.xml", "ppt/presentation.xml"])
def test_spreadsheet_alias_still_rejects_mixed_office_roots(root):
    source = _upload("xlsx")
    output = io.BytesIO(base64.b64decode(source["file_data"].partition(",")[2]))
    with zipfile.ZipFile(output, "a") as archive:
        archive.writestr(root, b"<document/>")
    with pytest.raises(DocumentError):
        preflight([part("xlsm", output.getvalue())], DoclingSettings())
