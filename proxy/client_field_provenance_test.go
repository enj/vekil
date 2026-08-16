package proxy

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Three separate leaks of client data into error_param have now come from the same shape: a
// name obtained by ranging over the CLIENT's own JSON object, handed to newChatInvalidRequest,
// which cannot tell where its param came from. Each was fixed by hand and the next one was
// written anyway, so this is the tripwire rather than a fourth round of care.
//
// The rule: if a name is bound by ranging over something that is not a literal vekil wrote, it
// is the client's, and it must go through newChatInvalidRequestClientField -- which keeps the
// full path for the client and records only the trusted parent. Ranging over a composite
// literal (`for _, f := range []string{"id", "status"}`) is vekil's own list and is fine.
func TestClientKeysDoNotReachNewChatInvalidRequest(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	checked := 0
	var offenders []string

	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		file, err := parser.ParseFile(fset, name, src, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		ast.Inspect(file, func(n ast.Node) bool {
			fn, ok := n.(*ast.FuncDecl)
			if !ok || fn.Body == nil {
				return true
			}
			// Names bound by ranging over something that is not a literal this package wrote.
			clientNames := map[string]bool{}
			ast.Inspect(fn.Body, func(inner ast.Node) bool {
				rng, ok := inner.(*ast.RangeStmt)
				if !ok {
					return true
				}
				if _, literal := rng.X.(*ast.CompositeLit); literal {
					return true
				}
				for _, target := range []ast.Expr{rng.Key, rng.Value} {
					if ident, ok := target.(*ast.Ident); ok && ident.Name != "_" {
						clientNames[ident.Name] = true
					}
				}
				return true
			})
			if len(clientNames) == 0 {
				return true
			}
			ast.Inspect(fn.Body, func(inner ast.Node) bool {
				call, ok := inner.(*ast.CallExpr)
				if !ok || len(call.Args) == 0 {
					return true
				}
				callee, ok := call.Fun.(*ast.Ident)
				if !ok || callee.Name != "newChatInvalidRequest" {
					return true
				}
				checked++
				// Only a name used AS the param, or concatenated into it, is a name. A range
				// variable reaching it any other way is an integer index inside a Sprintf --
				// "messages[%d]" is vekil's own structure, not the client's spelling. A client
				// key laundered through fmt.Sprintf("%s.%s", ...) would slip past this; no such
				// site exists today, and the sweep that added this checked for it.
				var suspects []*ast.Ident
				if ident, ok := call.Args[0].(*ast.Ident); ok {
					suspects = append(suspects, ident)
				}
				ast.Inspect(call.Args[0], func(arg ast.Node) bool {
					binary, ok := arg.(*ast.BinaryExpr)
					if !ok || binary.Op != token.ADD {
						return true
					}
					for _, side := range []ast.Expr{binary.X, binary.Y} {
						if ident, ok := side.(*ast.Ident); ok {
							suspects = append(suspects, ident)
						}
					}
					return true
				})
				for _, ident := range suspects {
					if clientNames[ident.Name] {
						offenders = append(offenders, fset.Position(call.Pos()).String()+
							": param derives from range variable "+ident.Name)
					}
				}
				return true
			})
			return true
		})
	}

	// Without this the test passes trivially the day someone renames the constructor.
	if checked == 0 {
		t.Fatal("scanned no newChatInvalidRequest call inside a range; this test no longer checks anything")
	}
	for _, offender := range offenders {
		t.Errorf("%s: use newChatInvalidRequestClientField so only the trusted parent is logged", offender)
	}
}
