import asyncio
import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class NabletFingerprintClient:
    def __init__(
        self,
        api_url: str = None,
        username: str = None,
        password: str = None,
        sdk_root: Optional[str] = None
    ):
        self.api_url = api_url or os.getenv("MEDIAENGINE_API_URL", "http://127.0.0.1:8090/api/v1.1")
        self.username = username or os.getenv("MEDIAENGINE_USERNAME", "admin")
        self.password = password or os.getenv("MEDIAENGINE_PASSWORD", "admin")
        
        sdk_path = sdk_root or os.getenv("MEDIAENGINE_SDK_PATH") or r"C:\Program Files\nablet\mediaEngine SDK\v3.2.180"
        self.sdk_root = Path(sdk_path)
        self.vars_script = self.sdk_root / "env" / "vars.ps1"
        self.rest_server_process = None
        
        if self.sdk_root.exists():
            logger.info(f"Using mediaEngine SDK at: {self.sdk_root}")
        else:
            logger.warning(f"mediaEngine SDK path not found: {self.sdk_root}")
        
        if self.vars_script.exists():
            logger.info(f"Found environment script: {self.vars_script}")
        else:
            logger.warning(f"Environment script not found: {self.vars_script}")
    
    def check_rest_server(self) -> bool:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 8090))
            sock.close()
            return result == 0
        except:
            return False
    
    def start_rest_server(self) -> bool:
        try:
            if self.check_rest_server():
                logger.info("REST server already running on port 8090")
                return True
            
            logger.info("Starting REST server...")
            
            start_cmd = f'''
                Set-Location "{self.sdk_root}"
                . "{self.vars_script}"
                Start-Process powershell -ArgumentList "-NoExit", "-Command", "rest_server --http -e http://127.0.0.1:8090 -p 8090 --http-access-log NUL:" -WindowStyle Hidden
            '''.strip()
            
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", start_cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            import time
            for i in range(30):
                time.sleep(0.5)
                if self.check_rest_server():
                    logger.info("REST server started successfully")
                    return True
            
            logger.error("REST server failed to start within 15 seconds")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start REST server: {e}")
            return False
    
    def stop_rest_server(self):
        try:
            logger.info("Stopping REST server...")
            
            stop_cmd = 'Get-Process | Where-Object {$_.ProcessName -eq "rest_server"} | Stop-Process -Force'
            
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", stop_cmd],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            import time
            for i in range(10):
                if not self.check_rest_server():
                    logger.info("REST server stopped")
                    return
                time.sleep(0.5)
            
            logger.warning("REST server may still be running")
            
        except Exception as e:
            logger.error(f"Error stopping REST server: {e}")
        
    def _get_powershell_command(self, command: str, args: List[str] = None) -> str:
        args = args or []
        
        ps_command = f'''
            Set-Location "{self.sdk_root}"
            . "{self.vars_script}"
            {command} {' '.join(args)}
        '''.strip()
        
        return ps_command
    
    def _run_rest_command(self, args: List[str]) -> Dict[str, Any]:
        try:
            ps_command = f'''
                cd "{self.sdk_root}";
                $env:MEDIAENGINE_API_URL = "{self.api_url}";
                $env:MEDIAENGINE_USERNAME = "{self.username}";
                $env:MEDIAENGINE_PASSWORD = "{self.password}";
                . "{self.vars_script}";
                rest {' '.join(args)}
            '''.strip()
            
            full_command = f"rest {' '.join(args)}"
            logger.info(f"Executing: {full_command}")
            logger.info(f"Working directory: {self.sdk_root}")
            logger.info(f"API URL: {self.api_url}")
            
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.sdk_root)
            )
            
            output = result.stdout.strip()
            stderr = result.stderr.strip()
            
            logger.info(f"Command completed with exit code: {result.returncode}")
            if output:
                logger.info(f"STDOUT:\n{output}")
            if stderr:
                logger.info(f"STDERR:\n{stderr}")
            
            # Nablet CLI writes REST errors to stderr, so check both streams
            combined_output = output + "\n" + stderr
            
            if "HTTP 501" in combined_output or "Not Implemented" in combined_output:
                raise Exception(f"REST command failed with HTTP 501: Not Implemented. Combined output: {combined_output[:300]}")
            
            if "HTTP 404" in combined_output or "Not Found" in combined_output:
                raise Exception(f"REST command failed with HTTP 404: Not Found. Combined output: {combined_output[:300]}")
            
            if "Connection refused" in combined_output or "Failed to connect" in combined_output:
                raise Exception(f"Cannot connect to REST server at {self.api_url}. Make sure REST server is running.")
            
            if "error:" in combined_output.lower():
                logger.error(f"Command failed with error in output")
                raise Exception(f"REST command failed: {combined_output[:500]}")
            
            if result.returncode != 0:
                logger.error(f"Command failed with non-zero exit code: {result.returncode}")
                raise Exception(f"REST command failed with exit code {result.returncode}. Output: {combined_output[:500]}")
            
            return {"success": True, "output": result.stdout, "stderr": result.stderr}
            
        except subprocess.TimeoutExpired:
            logger.error("Command timeout")
            raise Exception("Command timeout after 5 minutes")
        except FileNotFoundError as e:
            logger.error(f"PowerShell not found: {e}")
            raise Exception(f"PowerShell not found. Error: {e}")
        except Exception as e:
            logger.error(f"Command error: {e}")
            raise
    
    async def create_volume(self, volume_id: str, path: str, display_name: str) -> Dict[str, Any]:
        try:
            result = self._run_rest_command([
                "vadd",
                "--display-name", f'"{display_name}"',
                f"fp-{volume_id}",
                path
            ])
            logger.info(f"Volume created: fp-{volume_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to create volume: {e}")
            raise
    
    async def list_volumes(self) -> List[Dict[str, Any]]:
        try:
            result = self._run_rest_command(["vlist"])
            return result
        except Exception as e:
            logger.error(f"Failed to list volumes: {e}")
            raise
    
    async def create_project(self, display_name: str) -> str:
        try:
            result = self._run_rest_command(["padd", f'"{display_name}"'])
            
            output = result["output"]
            logger.info(f"Project creation output: {output}")
            
            for line in output.split("\n"):
                line_lower = line.lower()
                if "created" in line_lower or "project" in line_lower or "id" in line_lower:
                    logger.debug(f"Checking line: {line}")
                    parts = line.split()
                    for part in parts:
                        clean_part = part.strip('",;:')
                        if len(clean_part) == 24 and all(c in "0123456789abcdef" for c in clean_part.lower()):
                            logger.info(f"Extracted project ID: {clean_part}")
                            return clean_part
            
            import re
            hex_pattern = re.compile(r'\b([0-9a-fA-F]{20,32})\b')
            matches = hex_pattern.findall(output)
            if matches:
                project_id = matches[0]
                logger.info(f"Extracted project ID via regex: {project_id}")
                return project_id
            
            logger.error(f"Failed to extract project ID. Full output:\n{output}")
            logger.error(f"Full stderr:\n{result.get('stderr', 'N/A')}")
            
            raise Exception("Could not extract project ID from padd output")
            
        except Exception as e:
            error_msg = str(e)
            
            if "HTTP 409" in error_msg or "already exists" in error_msg.lower():
                logger.warning(f"Project '{display_name}' already exists. Fetching existing project ID...")
                
                try:
                    list_result = self._run_rest_command(["plist"])
                    output = list_result["output"]
                    
                    import re
                    for line in output.split("\n"):
                        if display_name in line:
                            hex_pattern = re.compile(r'\b([0-9a-fA-F]{20,32})\b')
                            matches = hex_pattern.findall(line)
                            if matches:
                                project_id = matches[0]
                                logger.info(f"Found existing project ID: {project_id}")
                                return project_id
                    
                    logger.error(f"Could not find project '{display_name}' in project list")
                    raise Exception(f"Project '{display_name}' exists but could not retrieve its ID")
                    
                except Exception as list_error:
                    logger.error(f"Failed to list projects: {list_error}")
                    raise Exception(f"Project already exists but could not retrieve ID: {list_error}")
            else:
                logger.error(f"Failed to create project: {e}")
                raise
    
    async def list_projects(self) -> List[Dict[str, Any]]:
        try:
            result = self._run_rest_command(["plist"])
            return result
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            raise
    
    async def generate_fpraw(
        self,
        source_path: str,
        output_path: str,
        parallel: int = 5
    ) -> Dict[str, Any]:
        try:
            source = Path(source_path).absolute()
            output = Path(output_path).absolute()
            output.mkdir(parents=True, exist_ok=True)
            
            log_file = Path.home() / "folderwatch_nablet.log"
            
            # Must run from SDK directory; generate_fpraw_cmd.yml is a relative path inside the SDK
            ps_command = f'''
                cd "{self.sdk_root}"
                . "{self.vars_script}"
                folderwatch -q --logfile "{log_file}" --delay 0 --parallel {parallel} -t 0 -i -E "(.*\\.(mxf|mp4|mov|ts|mkv|avi|webm|wmv|png)$)" --script .\\generate_fpraw_cmd.yml --var "output={output}" "{source}"
            '''.strip()
            
            logger.info(f"Generating FPRAW files from {source} to {output}")
            
            env = os.environ.copy()
            
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
                cwd=str(self.sdk_root)
            )
            
            output_text = result.stdout + "\n" + result.stderr
            logger.info(f"FPRAW generation completed. Output: {output_text}")
            
            created_files = list(output.rglob("*.*"))
            logger.info(f"Files created in {output}: {[f.name for f in created_files]}")
            
            if not created_files:
                logger.warning("No files were created by folderwatch")
            
            return {"success": True, "output": str(output), "stdout": output_text, "files_created": len(created_files)}
            
        except Exception as e:
            logger.error(f"Failed to generate FPRAW: {e}")
            raise
    
    async def add_media_to_project(
        self,
        project_id: str,
        volume_path: str
    ) -> Dict[str, Any]:
        try:
            result = self._run_rest_command([
                "add",
                "--copy", "none",
                "-R", project_id,
                volume_path
            ])
            logger.info(f"Media added to project {project_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to add media to project: {e}")
            raise
    
    async def generate_search_report(
        self,
        project_id: str,
        query_volume_path: str,
        display_name: str
    ) -> str:
        try:
            result = self._run_rest_command([
                "qadd",
                project_id,
                "--display-name", display_name,
                query_volume_path
            ])
            
            output = result["output"]
            logger.info(f"Search report output: {output}")
            
            for line in output.split("\n"):
                line_lower = line.lower()
                if "queued" in line_lower or "report" in line_lower or "created" in line_lower:
                    logger.debug(f"Checking line: {line}")
                    parts = line.split()
                    for part in parts:
                        clean_part = part.strip('",;:')
                        if len(clean_part) == 24 and all(c in "0123456789abcdef" for c in clean_part.lower()):
                            logger.info(f"Extracted report ID: {clean_part}")
                            return clean_part
            
            import re
            hex_pattern = re.compile(r'\b([0-9a-fA-F]{20,32})\b')
            matches = hex_pattern.findall(output)
            if matches:
                report_id = matches[0]
                logger.info(f"Extracted report ID via regex: {report_id}")
                return report_id
            
            logger.error(f"Failed to extract report ID. Full output:\n{output}")
            raise Exception(f"Could not extract report ID from output. Output was: {output[:200]}")
            
        except Exception as e:
            logger.error(f"Failed to generate search report: {e}")
            raise
    
    async def get_report_status(
        self,
        project_id: str,
        report_id: str
    ) -> Dict[str, Any]:
        try:
            result = self._run_rest_command([
                "qstatus",
                project_id,
                report_id
            ])
            
            status_info = {
                "status": "unknown",
                "progress": "0/0",
                "report_id": report_id
            }
            
            for line in result["output"].split("\n"):
                if "completed" in line.lower():
                    status_info["status"] = "completed"
                elif "processing" in line.lower() or "queued" in line.lower():
                    status_info["status"] = "processing"
                elif "failed" in line.lower():
                    status_info["status"] = "failed"
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get report status: {e}")
            raise
    
    async def get_report_json(
        self,
        project_id: str,
        report_id: str
    ) -> Dict[str, Any]:
        try:
            result = self._run_rest_command([
                "qreport",
                project_id,
                report_id
            ])
            
            output = result["output"]
            
            # The CLI output includes env setup text before the JSON payload
            try:
                json_start = output.find('{')
                if json_start != -1:
                    json_str = output[json_start:]
                    report_data = json.loads(json_str)
                    return report_data
                else:
                    return {"raw_output": output}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from output: {e}")
                return {"raw_output": output}
            
        except Exception as e:
            logger.error(f"Failed to get report JSON: {e}")
            raise
    
    async def list_reports(self, project_id: str) -> List[Dict[str, Any]]:
        try:
            result = self._run_rest_command([
                "qlist",
                project_id
            ])
            return result
        except Exception as e:
            logger.error(f"Failed to list reports: {e}")
            raise
