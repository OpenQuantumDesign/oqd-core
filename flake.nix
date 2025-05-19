# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
{
  description = "OpenQuantum Design Core Library";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    oqd-compiler-infrastructure = {
      url = "github:OpenQuantumDesign/oqd-compiler-infrastructure/Mahmoud-Yasser-18-patch-1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, oqd-compiler-infrastructure }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        pythonPackages = python.pkgs;
      in
      {
        packages.default = pythonPackages.callPackage ./pkgs/derivation.nix { };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Python with dependencies
            (python.withPackages (ps: with ps; [
              # Core dependencies
              pydantic
              numpy

              # Documentation dependencies
              pymdown-extensions
              mkdocstrings
              mkdocs-material
              mkdocstrings-python
              mdx-truly-sane-lists

              # Test dependencies
              pytest

              # Development tools
              pip
              black
              isort
              mypy
            ]))

            # Additional development tools
            git
          ];

          shellHook = ''
            export PYTHONPATH="$PWD/src:$PYTHONPATH"
            echo "OpenQuantum Design Core Library development environment"
            echo "Python version: $(python --version)"
          '';
        };
      }
    );
}
