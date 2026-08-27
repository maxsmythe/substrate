// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"cloud.google.com/go/iam/apiv1/iampb"
	resourcemanager "cloud.google.com/go/resourcemanager/apiv3"
	"cloud.google.com/go/resourcemanager/apiv3/resourcemanagerpb"
	serviceusage "cloud.google.com/go/serviceusage/apiv1"
	"cloud.google.com/go/serviceusage/apiv1/serviceusagepb"
	"github.com/spf13/cobra"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	iam "google.golang.org/api/iam/v1"
	oauth2v2 "google.golang.org/api/oauth2/v2"
	"google.golang.org/api/option"
	servicenetworking "google.golang.org/api/servicenetworking/v1"
	sqladmin "google.golang.org/api/sqladmin/v1"
)

// The ateapi store database and the Kubernetes identity that connects to it.
// These match the atepg store (hack/install-ate.sh) and the ate-api-server
// deployment.
const (
	cloudSQLDatabase   = "atepg"
	apiServerNamespace = "ate-system"
	apiServerKSA       = "ate-api-server"
)

func cloudSQLGSAEmail(cfg *Config) string {
	return fmt.Sprintf("%s@%s.iam.gserviceaccount.com", cfg.CloudSQLGSAName, cfg.ProjectID)
}

// cloudSQLDatabaseUser derives the IAM database username for a service
// account: the email with the .gserviceaccount.com suffix trimmed.
func cloudSQLDatabaseUser(gsaEmail string) string {
	return strings.TrimSuffix(gsaEmail, ".gserviceaccount.com")
}

// workloadIdentityMember is the classic Workload Identity member for the
// ate-api-server KSA. WIF-direct principal:// identities (used elsewhere in
// this tool) cannot be Cloud SQL IAM database users, so Cloud SQL requires
// the KSA→GSA link.
func workloadIdentityMember(projectID string) string {
	return fmt.Sprintf("serviceAccount:%s.svc.id.goog[%s/%s]", projectID, apiServerNamespace, apiServerKSA)
}

func privateNetworkURL(cfg *Config) string {
	return fmt.Sprintf("projects/%s/global/networks/%s", cfg.ProjectID, cfg.Network)
}

func isNotFound(err error) bool {
	var gerr *googleapi.Error
	return errors.As(err, &gerr) && gerr.Code == 404
}

// ensureCallerSAAdmin grants roles/iam.serviceAccountAdmin at project
// scope to the caller so that the later SetIamPolicy on the ate-api-server
// GSA (Workload Identity binding) succeeds. Idempotent. A per-SA grant
// would be tighter, but setting a per-SA policy needs
// iam.serviceAccounts.setIamPolicy — the exact permission this bootstraps.
// Project IAM Admin (which Cloud SQL operators tend to already have) is
// enough to add the grant here.
func ensureCallerSAAdmin(ctx context.Context, cfg *Config) error {
	member, err := resolveCallerMember(ctx, cfg)
	if err != nil {
		return err
	}
	client, err := resourcemanager.NewProjectsClient(ctx)
	if err != nil {
		return err
	}
	defer client.Close()
	resource := fmt.Sprintf("projects/%s", cfg.ProjectID)
	policy, err := client.GetIamPolicy(ctx, &iampb.GetIamPolicyRequest{Resource: resource})
	if err != nil {
		return fmt.Errorf("get project iam policy: %w", err)
	}
	if !addProjectIamBinding(policy, "roles/iam.serviceAccountAdmin", member) {
		slog.Info("Caller already has roles/iam.serviceAccountAdmin. Skipping.",
			slog.String("member", member), slog.String("project", cfg.ProjectID))
		return nil
	}
	slog.Info("Granting caller roles/iam.serviceAccountAdmin at project scope...",
		slog.String("member", member), slog.String("project", cfg.ProjectID))
	if _, err := client.SetIamPolicy(ctx, &iampb.SetIamPolicyRequest{Resource: resource, Policy: policy}); err != nil {
		return fmt.Errorf("set project iam policy: %w", err)
	}
	return nil
}

// resolveCallerMember returns the IAM member string for the identity
// running the command. cfg.CloudSQLCaller wins if set; otherwise the
// OAuth2 userinfo endpoint identifies the ADC subject. Emails ending in
// gserviceaccount.com become serviceAccount:<email>; everything else is
// user:<email>.
func resolveCallerMember(ctx context.Context, cfg *Config) (string, error) {
	if cfg.CloudSQLCaller != "" {
		return cfg.CloudSQLCaller, nil
	}
	svc, err := oauth2v2.NewService(ctx)
	if err != nil {
		return "", fmt.Errorf("create oauth2 client (pass --caller to override): %w", err)
	}
	info, err := svc.Userinfo.Get().Context(ctx).Do()
	if err != nil {
		return "", fmt.Errorf("resolve caller identity via oauth2 userinfo (pass --caller=<user:email|serviceAccount:email> to override): %w", err)
	}
	if info.Email == "" {
		return "", errors.New("caller identity has no email; pass --caller=<user:email|serviceAccount:email>")
	}
	prefix := "user:"
	if strings.HasSuffix(info.Email, "gserviceaccount.com") {
		prefix = "serviceAccount:"
	}
	return prefix + info.Email, nil
}

// enableCloudSQLAPIs idempotently enables the APIs this command depends on.
// Scoped here rather than in `enable apis`: Cloud SQL is opt-in, so only its
// users get these APIs turned on.
func enableCloudSQLAPIs(ctx context.Context, cfg *Config) error {
	suClient, err := serviceusage.NewClient(ctx)
	if err != nil {
		return err
	}
	defer suClient.Close()

	services := []string{
		"sqladmin.googleapis.com",
		"servicenetworking.googleapis.com",
		"compute.googleapis.com",
	}
	slog.Info("Batch enabling services", slog.String("services", strings.Join(services, ", ")), slog.String("project", cfg.ProjectID))
	op, err := suClient.BatchEnableServices(ctx, &serviceusagepb.BatchEnableServicesRequest{
		Parent:     fmt.Sprintf("projects/%s", cfg.ProjectID),
		ServiceIds: services,
	})
	if err != nil {
		return fmt.Errorf("failed to start batch enabling services: %w", err)
	}
	if _, err := op.Wait(ctx); err != nil {
		return fmt.Errorf("failed to complete batch enabling services: %w", err)
	}
	return nil
}

// ensurePrivateServicesAccess makes sure the VPC has a private services
// access peering (Cloud SQL private IP requires it). It reserves a /16
// range and creates the servicenetworking peering if either is missing.
// Both operations are per-VPC and idempotent.
func ensurePrivateServicesAccess(ctx context.Context, cfg *Config) error {
	snSvc, err := servicenetworking.NewService(ctx)
	if err != nil {
		return fmt.Errorf("create servicenetworking client: %w", err)
	}
	resp, err := snSvc.Services.Connections.List("services/servicenetworking.googleapis.com").
		Network(privateNetworkURL(cfg)).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("list service networking connections: %w", err)
	}
	for _, c := range resp.Connections {
		if len(c.ReservedPeeringRanges) > 0 {
			slog.Info("Private services access peering already configured. Skipping.",
				slog.String("network", cfg.Network),
				slog.String("ranges", strings.Join(c.ReservedPeeringRanges, ",")))
			return nil
		}
	}

	rangeName := "google-managed-services-" + cfg.Network
	if err := ensurePeeringRange(ctx, cfg, rangeName); err != nil {
		return err
	}

	projectNumber, err := resolveProjectNumber(ctx, cfg)
	if err != nil {
		return fmt.Errorf("resolve project number: %w", err)
	}
	slog.Info("Creating service networking connection (VPC peering)...",
		slog.String("network", cfg.Network), slog.String("range", rangeName))
	// The connection body requires the network URL with project number, not
	// project ID; the address reservation above uses project ID because the
	// compute API accepts either.
	op, err := snSvc.Services.Connections.Create("services/servicenetworking.googleapis.com",
		&servicenetworking.Connection{
			Network:               fmt.Sprintf("projects/%s/global/networks/%s", projectNumber, cfg.Network),
			ReservedPeeringRanges: []string{rangeName},
		}).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("create service networking connection: %w", err)
	}
	return waitForServiceNetworkingOperation(ctx, snSvc, op)
}

// ensurePeeringRange reserves the PSA range at a deterministic, GKE-safe
// address (default 192.168.0.0/16). Auto-allocation would pick from
// 10.0.0.0/8, which routinely collides with GKE's auto-allocated pod
// secondary range on large clusters — the resulting overlap silently
// breaks VPC peering's route advertisement, so pods lose the Cloud SQL
// return path and every psql call hangs.
func ensurePeeringRange(ctx context.Context, cfg *Config, rangeName string) error {
	svc, err := compute.NewService(ctx)
	if err != nil {
		return fmt.Errorf("create compute client: %w", err)
	}
	address, prefixLength, err := parsePSARange(cfg.CloudSQLPSARange)
	if err != nil {
		return err
	}
	existing, err := svc.GlobalAddresses.Get(cfg.ProjectID, rangeName).Context(ctx).Do()
	if err == nil {
		if existing.Address != address || existing.PrefixLength != prefixLength {
			slog.Warn("VPC peering range exists with a different range than requested; skipping create. Recreate it manually if pods can't reach Cloud SQL (pod-CIDR overlap risk when the existing range is inside 10.0.0.0/8).",
				slog.String("range", rangeName),
				slog.String("existing", fmt.Sprintf("%s/%d", existing.Address, existing.PrefixLength)),
				slog.String("requested", fmt.Sprintf("%s/%d", address, prefixLength)))
		} else {
			slog.Info("VPC peering range exists. Skipping create.", slog.String("range", rangeName))
		}
		return nil
	}
	if !isNotFound(err) {
		return fmt.Errorf("get global address: %w", err)
	}
	slog.Info("Reserving VPC peering range...",
		slog.String("range", rangeName), slog.String("network", cfg.Network),
		slog.String("address", fmt.Sprintf("%s/%d", address, prefixLength)))
	op, err := svc.GlobalAddresses.Insert(cfg.ProjectID, &compute.Address{
		Name:         rangeName,
		Purpose:      "VPC_PEERING",
		AddressType:  "INTERNAL",
		Address:      address,
		PrefixLength: prefixLength,
		Network:      privateNetworkURL(cfg),
	}).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("reserve global address: %w", err)
	}
	return waitForGlobalComputeOperation(ctx, svc, cfg, op)
}

// parsePSARange splits a CIDR like "192.168.0.0/16" into its address and
// prefix length for the compute Address insert body.
func parsePSARange(cidr string) (string, int64, error) {
	slash := strings.LastIndex(cidr, "/")
	if slash < 0 {
		return "", 0, fmt.Errorf("--psa-range %q: want <address>/<prefix>, e.g. 192.168.0.0/16", cidr)
	}
	prefix, err := strconv.ParseInt(cidr[slash+1:], 10, 64)
	if err != nil || prefix < 8 || prefix > 29 {
		return "", 0, fmt.Errorf("--psa-range %q: prefix must be 8..29", cidr)
	}
	return cidr[:slash], prefix, nil
}

func waitForGlobalComputeOperation(ctx context.Context, svc *compute.Service, cfg *Config, op *compute.Operation) error {
	name := op.Name
	for {
		if op.Status == "DONE" {
			if op.Error != nil && len(op.Error.Errors) > 0 {
				return fmt.Errorf("operation %s failed: %s", name, op.Error.Errors[0].Message)
			}
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(3 * time.Second):
		}
		var err error
		op, err = svc.GlobalOperations.Get(cfg.ProjectID, name).Context(ctx).Do()
		if err != nil {
			return fmt.Errorf("poll operation %s: %w", name, err)
		}
	}
}

func waitForServiceNetworkingOperation(ctx context.Context, svc *servicenetworking.APIService, op *servicenetworking.Operation) error {
	name := op.Name
	for {
		if op.Done {
			if op.Error != nil {
				return fmt.Errorf("operation %s failed: %s", name, op.Error.Message)
			}
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
		var err error
		op, err = svc.Operations.Get(name).Context(ctx).Do()
		if err != nil {
			// op is nil here on failure; use the cached name.
			return fmt.Errorf("poll operation %s: %w", name, err)
		}
	}
}

// resolveProjectNumber returns cfg.ProjectNumber if set, otherwise looks it
// up from the project ID. The servicenetworking Connection API requires the
// network URL to use project number.
func resolveProjectNumber(ctx context.Context, cfg *Config) (string, error) {
	if cfg.ProjectNumber != "" {
		return cfg.ProjectNumber, nil
	}
	client, err := resourcemanager.NewProjectsClient(ctx)
	if err != nil {
		return "", err
	}
	defer client.Close()
	proj, err := client.GetProject(ctx, &resourcemanagerpb.GetProjectRequest{
		Name: "projects/" + cfg.ProjectID,
	})
	if err != nil {
		return "", fmt.Errorf("get project: %w", err)
	}
	// proj.Name is "projects/<number>".
	return strings.TrimPrefix(proj.Name, "projects/"), nil
}

func waitForSQLOperation(ctx context.Context, svc *sqladmin.Service, cfg *Config, op *sqladmin.Operation) error {
	name := op.Name
	for {
		if op.Status == "DONE" {
			if op.Error != nil && len(op.Error.Errors) > 0 {
				return fmt.Errorf("operation %s failed: %s", name, op.Error.Errors[0].Message)
			}
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
		var err error
		op, err = svc.Operations.Get(cfg.ProjectID, name).Context(ctx).Do()
		if err != nil {
			return fmt.Errorf("poll operation %s: %w", name, err)
		}
	}
}

// createCloudSQLInstance creates a private-IP PostgreSQL instance with IAM
// database authentication enabled, or verifies an existing one.
func createCloudSQLInstance(ctx context.Context, svc *sqladmin.Service, cfg *Config) error {
	slog.Info("Checking if Cloud SQL instance exists", slog.String("instance", cfg.CloudSQLInstance), slog.String("project", cfg.ProjectID))
	existing, err := svc.Instances.Get(cfg.ProjectID, cfg.CloudSQLInstance).Context(ctx).Do()
	if err == nil {
		iamAuthOn := false
		if existing.Settings != nil {
			for _, f := range existing.Settings.DatabaseFlags {
				if f.Name == "cloudsql.iam_authentication" && f.Value == "on" {
					iamAuthOn = true
				}
			}
		}
		if !iamAuthOn {
			return fmt.Errorf("instance %s exists but cloudsql.iam_authentication is off; enable it (this restarts the instance):\n\n  gcloud sql instances patch %s --database-flags=cloudsql.iam_authentication=on --project=%s",
				cfg.CloudSQLInstance, cfg.CloudSQLInstance, cfg.ProjectID)
		}
		slog.Info("Cloud SQL instance exists with IAM authentication on. Skipping create.")
		return nil
	}
	if !isNotFound(err) {
		return fmt.Errorf("get instance: %w", err)
	}

	if err := ensurePrivateServicesAccess(ctx, cfg); err != nil {
		return err
	}

	spec, err := cloudSQLInstanceSpec(cfg)
	if err != nil {
		return err
	}
	slog.Info("Creating Cloud SQL instance (takes several minutes)...",
		slog.String("instance", cfg.CloudSQLInstance), slog.String("tier", cfg.CloudSQLTier),
		slog.String("edition", spec.Settings.Edition))
	op, err := svc.Instances.Insert(cfg.ProjectID, spec).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("create instance: %w", err)
	}
	return waitForSQLOperation(ctx, svc, cfg, op)
}

// cloudSQLInstanceSpec maps the flags onto the creation request. Pure, for
// testing. Shape/storage of an already-existing instance is never touched —
// resize with `gcloud sql instances patch`.
func cloudSQLInstanceSpec(cfg *Config) (*sqladmin.DatabaseInstance, error) {
	settings := &sqladmin.Settings{
		Tier: cfg.CloudSQLTier,
		IpConfiguration: &sqladmin.IpConfiguration{
			Ipv4Enabled:     false,
			PrivateNetwork:  privateNetworkURL(cfg),
			ForceSendFields: []string{"Ipv4Enabled"},
		},
		DatabaseFlags: []*sqladmin.DatabaseFlags{
			{Name: "cloudsql.iam_authentication", Value: "on"},
		},
	}
	switch cfg.CloudSQLEdition {
	case "", "enterprise":
		// Enterprise edition accepts db-custom-<vCPU>-<MB> tiers.
		settings.Edition = "ENTERPRISE"
	case "enterprise-plus":
		// Enterprise Plus only accepts db-perf-optimized-N-<vCPU> tiers. Its
		// local-SSD data cache is the reason to pick it for this store: it
		// extends the effective cache beyond RAM once the dataset outgrows
		// memory (see cloud-sql.md, "Scaling the database").
		settings.Edition = "ENTERPRISE_PLUS"
		settings.DataCacheConfig = &sqladmin.DataCacheConfig{DataCacheEnabled: true}
	default:
		return nil, fmt.Errorf("unknown --edition %q (want enterprise|enterprise-plus)", cfg.CloudSQLEdition)
	}
	// 0 leaves the Cloud SQL default (10 GB, auto-resizing). PD IOPS and
	// throughput scale with provisioned size, so benchmarks and production
	// should pre-size rather than rely on auto-resize.
	if cfg.CloudSQLStorageGB > 0 {
		settings.DataDiskSizeGb = cfg.CloudSQLStorageGB
	}
	return &sqladmin.DatabaseInstance{
		Name:            cfg.CloudSQLInstance,
		Region:          cfg.Region,
		DatabaseVersion: "POSTGRES_18",
		Settings:        settings,
	}, nil
}

func createCloudSQLDatabase(ctx context.Context, svc *sqladmin.Service, cfg *Config) error {
	_, err := svc.Databases.Get(cfg.ProjectID, cfg.CloudSQLInstance, cloudSQLDatabase).Context(ctx).Do()
	if err == nil {
		slog.Info("Database exists. Skipping create.", slog.String("database", cloudSQLDatabase))
		return nil
	}
	if !isNotFound(err) {
		return fmt.Errorf("get database: %w", err)
	}
	slog.Info("Creating database...", slog.String("database", cloudSQLDatabase))
	op, err := svc.Databases.Insert(cfg.ProjectID, cfg.CloudSQLInstance, &sqladmin.Database{
		Name: cloudSQLDatabase,
	}).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("create database: %w", err)
	}
	return waitForSQLOperation(ctx, svc, cfg, op)
}

func createCloudSQLGSA(ctx context.Context, cfg *Config) error {
	svc, err := iam.NewService(ctx)
	if err != nil {
		return fmt.Errorf("create iam client: %w", err)
	}
	email := cloudSQLGSAEmail(cfg)
	resource := fmt.Sprintf("projects/%s/serviceAccounts/%s", cfg.ProjectID, email)
	_, err = svc.Projects.ServiceAccounts.Get(resource).Context(ctx).Do()
	if err == nil {
		slog.Info("Service account exists. Skipping create.", slog.String("gsa", email))
	} else if isNotFound(err) {
		slog.Info("Creating service account...", slog.String("gsa", email))
		_, err = svc.Projects.ServiceAccounts.Create("projects/"+cfg.ProjectID, &iam.CreateServiceAccountRequest{
			AccountId: cfg.CloudSQLGSAName,
			ServiceAccount: &iam.ServiceAccount{
				DisplayName: "ate-api-server Cloud SQL access",
			},
		}).Context(ctx).Do()
		if err != nil {
			return fmt.Errorf("create service account: %w", err)
		}
	} else {
		return fmt.Errorf("get service account: %w", err)
	}

	// Classic Workload Identity: let the ate-api-server KSA mint tokens as
	// this GSA, so the proxy sidecar's ADC resolves to the IAM database user.
	member := workloadIdentityMember(cfg.ProjectID)
	policy, err := svc.Projects.ServiceAccounts.GetIamPolicy(resource).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("get service account iam policy: %w", err)
	}
	const wiRole = "roles/iam.workloadIdentityUser"
	for _, b := range policy.Bindings {
		if b.Role == wiRole {
			for _, m := range b.Members {
				if m == member {
					slog.Info("Workload Identity binding exists. Skipping.")
					return nil
				}
			}
			b.Members = append(b.Members, member)
			return setGSAPolicy(ctx, svc, resource, policy)
		}
	}
	policy.Bindings = append(policy.Bindings, &iam.Binding{Role: wiRole, Members: []string{member}})
	return setGSAPolicy(ctx, svc, resource, policy)
}

func setGSAPolicy(ctx context.Context, svc *iam.Service, resource string, policy *iam.Policy) error {
	slog.Info("Binding Workload Identity user to service account...")
	_, err := svc.Projects.ServiceAccounts.SetIamPolicy(resource, &iam.SetIamPolicyRequest{Policy: policy}).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("set service account iam policy: %w", err)
	}
	return nil
}

// grantCloudSQLProjectRoles grants the GSA the roles needed to establish
// connector tunnels (cloudsql.client) and to log in with IAM database
// authentication (cloudsql.instanceUser).
func grantCloudSQLProjectRoles(ctx context.Context, cfg *Config) error {
	client, err := resourcemanager.NewProjectsClient(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	resource := fmt.Sprintf("projects/%s", cfg.ProjectID)
	policy, err := client.GetIamPolicy(ctx, &iampb.GetIamPolicyRequest{Resource: resource})
	if err != nil {
		return fmt.Errorf("get project iam policy: %w", err)
	}

	member := "serviceAccount:" + cloudSQLGSAEmail(cfg)
	// A freshly created service account can take a while to propagate;
	// granting to it too soon fails with "does not exist". Re-read the
	// policy each attempt so etag conflicts also resolve.
	for attempt := 1; ; attempt++ {
		changed1 := addProjectIamBinding(policy, "roles/cloudsql.client", member)
		changed2 := addProjectIamBinding(policy, "roles/cloudsql.instanceUser", member)
		if !changed1 && !changed2 {
			slog.Info("IAM policy already has required Cloud SQL permissions. Skipping update.")
			return nil
		}
		slog.Info("Setting IAM policy (grant Cloud SQL permissions)...", slog.String("member", member))
		_, err = client.SetIamPolicy(ctx, &iampb.SetIamPolicyRequest{Resource: resource, Policy: policy})
		if err == nil {
			return nil
		}
		if attempt >= 6 {
			return fmt.Errorf("set project iam policy: %w", err)
		}
		slog.Warn("Setting IAM policy failed, retrying...", slog.Int("attempt", attempt), slog.Any("err", err))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(10 * time.Second):
		}
		policy, err = client.GetIamPolicy(ctx, &iampb.GetIamPolicyRequest{Resource: resource})
		if err != nil {
			return fmt.Errorf("get project iam policy: %w", err)
		}
	}
}

func createCloudSQLIAMUser(ctx context.Context, svc *sqladmin.Service, cfg *Config) error {
	dbUser := cloudSQLDatabaseUser(cloudSQLGSAEmail(cfg))
	users, err := svc.Users.List(cfg.ProjectID, cfg.CloudSQLInstance).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("list database users: %w", err)
	}
	for _, u := range users.Items {
		if u.Name == dbUser {
			slog.Info("IAM database user exists. Skipping create.", slog.String("user", dbUser))
			return nil
		}
	}
	slog.Info("Creating IAM database user...", slog.String("user", dbUser))
	// The API requires the truncated form (without .gserviceaccount.com),
	// which is also the username the database sees.
	op, err := svc.Users.Insert(cfg.ProjectID, cfg.CloudSQLInstance, &sqladmin.User{
		Name: dbUser,
		Type: "CLOUD_IAM_SERVICE_ACCOUNT",
	}).Context(ctx).Do()
	if err != nil {
		return fmt.Errorf("create IAM database user: %w", err)
	}
	return waitForSQLOperation(ctx, svc, cfg, op)
}

func printCloudSQLNextSteps(cfg *Config) {
	gsa := cloudSQLGSAEmail(cfg)
	dbUser := cloudSQLDatabaseUser(gsa)
	fmt.Printf(`
Cloud SQL is provisioned. Two steps remain:

1. One-time schema privileges (PostgreSQL 15+ removed PUBLIC's CREATE on the
   public schema; ateapi applies its schema at startup as the IAM user).
   Connect as the postgres user and run:

     GRANT USAGE, CREATE ON SCHEMA public TO "%s";

2. Deploy ateapi against it:

     export ATE_API_POSTGRES_CLOUDSQL_INSTANCE=%s:%s:%s
     export ATE_API_POSTGRES_CLOUDSQL_GSA=%s
     ./hack/install-ate.sh --deploy-ate-system

See tools/setup-gcp/cloud-sql.md for details and verification steps.
`, dbUser, cfg.ProjectID, cfg.Region, cfg.CloudSQLInstance, gsa)
}

var cloudsqlCmd = &cobra.Command{
	Use:   "cloudsql",
	Short: "Create a Cloud SQL PostgreSQL instance for the ateapi store, with IAM database authentication",
	RunE: func(cmd *cobra.Command, args []string) error {
		if cfg.ProjectID == "" {
			return errors.New("--project-id is required")
		}
		ctx := cmd.Context()
		if err := ensureCallerSAAdmin(ctx, &cfg); err != nil {
			return err
		}
		if err := enableCloudSQLAPIs(ctx, &cfg); err != nil {
			return err
		}
		svc, err := sqladmin.NewService(ctx, option.WithQuotaProject(cfg.ProjectID))
		if err != nil {
			return fmt.Errorf("create sqladmin client: %w", err)
		}
		if err := createCloudSQLInstance(ctx, svc, &cfg); err != nil {
			return err
		}
		if err := createCloudSQLDatabase(ctx, svc, &cfg); err != nil {
			return err
		}
		if err := createCloudSQLGSA(ctx, &cfg); err != nil {
			return err
		}
		if err := grantCloudSQLProjectRoles(ctx, &cfg); err != nil {
			return err
		}
		if err := createCloudSQLIAMUser(ctx, svc, &cfg); err != nil {
			return err
		}
		printCloudSQLNextSteps(&cfg)
		return nil
	},
}

func init() {
	createCmd.AddCommand(cloudsqlCmd)
	cloudsqlCmd.Flags().StringVar(&cfg.CloudSQLInstance, "instance", getEnv("CLOUDSQL_INSTANCE", "atepg"), "Cloud SQL instance name [env: CLOUDSQL_INSTANCE]")
	cloudsqlCmd.Flags().StringVar(&cfg.CloudSQLTier, "tier", getEnv("CLOUDSQL_TIER", "db-custom-2-8192"), "Machine tier: db-custom-<vCPU>-<MB> for enterprise, db-perf-optimized-N-<vCPU> for enterprise-plus [env: CLOUDSQL_TIER]")
	cloudsqlCmd.Flags().StringVar(&cfg.CloudSQLEdition, "edition", getEnv("CLOUDSQL_EDITION", "enterprise"), "Instance edition: enterprise | enterprise-plus (enables the local-SSD data cache) [env: CLOUDSQL_EDITION]")
	cloudsqlCmd.Flags().Int64Var(&cfg.CloudSQLStorageGB, "storage-size", getEnv("CLOUDSQL_STORAGE_GB", int64(0)), "Data disk size in GB; 0 = Cloud SQL default (10 GB, auto-resizing). PD IOPS scale with size [env: CLOUDSQL_STORAGE_GB]")
	cloudsqlCmd.Flags().StringVar(&cfg.CloudSQLGSAName, "gsa-name", getEnv("CLOUDSQL_GSA_NAME", "ate-api-server"), "Name of the Google service account to create for Workload Identity + IAM database auth [env: CLOUDSQL_GSA_NAME]")
	cloudsqlCmd.Flags().StringVar(&cfg.Network, "network", getEnv("NETWORK", "default"), "VPC network name (must match the cluster's) [env: NETWORK]")
	cloudsqlCmd.Flags().StringVar(&cfg.CloudSQLCaller, "caller", getEnv("CLOUDSQL_CALLER", ""), "IAM member for the calling identity (user:<email>, serviceAccount:<email>). Auto-detected from ADC if empty; only needed when userinfo lookup fails [env: CLOUDSQL_CALLER]")
	cloudsqlCmd.Flags().StringVar(&cfg.CloudSQLPSARange, "psa-range", getEnv("CLOUDSQL_PSA_RANGE", "192.168.0.0/16"), "CIDR to reserve for private services access (VPC peering). Default lives outside 10.0.0.0/8 to avoid overlap with GKE pod CIDR auto-allocation. Ignored if the peering already exists [env: CLOUDSQL_PSA_RANGE]")
}
